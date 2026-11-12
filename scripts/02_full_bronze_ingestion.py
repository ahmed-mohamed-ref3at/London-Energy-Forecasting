
import os
import sys
import csv
import json
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw" / "csv"
BRONZE_DIR = BASE_DIR / "data" / "bronze" / "readings_by_file"
STAGING_DIR = BASE_DIR / "data" / "bronze" / "_staging"
AUDIT_PATH = BASE_DIR / "reports" / "raw_audit_batched.csv"

BRONZE_DIR.mkdir(parents=True, exist_ok=True)
STAGING_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 5

# First run: process one new file.
# After successful test: change to None.
#MAX_NEW_FILES = 1
MAX_NEW_FILES = None


# Bronze flags mapped to original audit metrics
FLAG_MAP = {
    "missing_household_id": "is_missing_household_id",
    "missing_timestamp": "is_missing_timestamp",
    "invalid_timestamp": "is_invalid_timestamp",
    "missing_kwh": "is_missing_kwh",
    "invalid_kwh": "is_invalid_kwh",
    "negative_kwh": "is_negative_kwh",
}

METRICS = ["row_count", *FLAG_MAP.keys()]


# ----------------------------------------
# 1. Load and validate raw audit
# ----------------------------------------

audit = {}

with AUDIT_PATH.open(
    "r", newline="", encoding="utf-8-sig"
) as file:

    for row in csv.DictReader(file):

        name = row["source_file"]

        if name in audit:
            raise ValueError(
                f"Duplicate source in audit: {name}"
            )

        audit[name] = {
            key: int(row[key])
            for key in METRICS
        }


csv_files = sorted(RAW_DIR.rglob("*.csv"))

if len(csv_files) != 168 or len(audit) != 168:
    raise ValueError(
        "Expected 168 CSV files and 168 audit records."
    )

if {p.name for p in csv_files} != set(audit):
    raise ValueError(
        "Source CSV files do not match raw audit."
    )


# ----------------------------------------
# 2. Check saved progress
# ----------------------------------------

def final_path(source):
    return BRONZE_DIR / source.stem


def is_completed(source):

    destination = final_path(source)

    if not destination.exists():
        return False

    marker = destination / "_bronze_complete.json"
    success = destination / "_SUCCESS"

    if not marker.is_file() or not success.is_file():
        raise RuntimeError(
            f"Incomplete Bronze output: {destination}"
        )

    metadata = json.loads(
        marker.read_text(encoding="utf-8")
    )

    if (
        metadata["source_file"] != source.name
        or metadata["metrics"] != audit[source.name]
    ):
        raise ValueError(
            f"Checkpoint mismatch: {source.name}"
        )

    return True


pending_files = [
    p for p in csv_files
    if not is_completed(p)
]

print("=" * 55)
print("LONDON SMART METER - FULL BRONZE INGESTION")
print("=" * 55)

print("Total source files:", len(csv_files))
print("Already completed:", len(csv_files) - len(pending_files))
print("Remaining:", len(pending_files))


# ----------------------------------------
# 3. Spark setup
# ----------------------------------------

spark = (
    SparkSession.builder
    .appName("LondonEnergyFullBronze")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.timestampType", "TIMESTAMP_NTZ")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def is_missing(name):

    column = F.col(name)

    return (
        column.isNull()
        | (column == "")
        | (F.lower(column) == "null")
    )


def build_bronze(source):

    raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("mode", "PERMISSIVE")
        .csv(str(source))
    )

    raw = raw.toDF(
        *[name.strip() for name in raw.columns]
    )

    expected_columns = {
        "LCLid",
        "stdorToU",
        "DateTime",
        "KWH/hh (per half hour)",
    }

    if set(raw.columns) != expected_columns:
        raise ValueError(
            f"Unexpected columns in {source.name}"
        )

    df = raw.select(
        F.lit(source.name).alias("source_file"),
        F.col("LCLid").alias("raw_household_id"),
        F.col("stdorToU").alias("raw_tariff_group"),
        F.col("DateTime").alias("raw_datetime"),
        F.col("KWH/hh (per half hour)").alias("raw_kwh"),
    )

    df = (
        df
        .withColumn(
            "household_id",
            F.trim("raw_household_id")
        )
        .withColumn(
            "tariff_group",
            F.trim("raw_tariff_group")
        )
        .withColumn(
            "timestamp_text",
            F.trim("raw_datetime")
        )
        .withColumn(
            "kwh_text",
            F.trim("raw_kwh")
        )
        .withColumn(
            "reading_timestamp",
            F.try_to_timestamp(
                F.col("timestamp_text"),
                F.lit("yyyy-MM-dd HH:mm:ss.SSSSSSS")
            )
        )
    )

    numeric_pattern = (
        r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][+-]?\d+)?$"
    )

    df = df.withColumn(
        "kwh",
        F.when(
            F.col("kwh_text").rlike(numeric_pattern),
            F.col("kwh_text").cast("double")
        ).otherwise(F.lit(None).cast("double"))
    )

    df = (
        df
        .withColumn(
            "is_missing_household_id",
            is_missing("household_id")
        )
        .withColumn(
            "is_missing_timestamp",
            is_missing("timestamp_text")
        )
        .withColumn(
            "is_invalid_timestamp",
            ~is_missing("timestamp_text")
            & F.col("reading_timestamp").isNull()
        )
        .withColumn(
            "is_missing_kwh",
            is_missing("kwh_text")
        )
        .withColumn(
            "is_invalid_kwh",
            ~is_missing("kwh_text")
            & F.col("kwh").isNull()
        )
        .withColumn(
            "is_negative_kwh",
            F.coalesce(
                F.col("kwh") < 0,
                F.lit(False)
            )
        )
    )

    return df


# ----------------------------------------
# 4. Process and validate one file
# ----------------------------------------

def process_file(source):

    destination = final_path(source)
    staging = STAGING_DIR / source.stem

    if staging.exists():
        raise RuntimeError(
            "Unfinished staging directory found: "
            f"{staging}. Inspect it before retrying."
        )

    print(f"\nProcessing: {source.name}", flush=True)

    df = build_bronze(source)

    # Write to a temporary directory first.
    df.write.mode("errorifexists").parquet(str(staging))

    # Read back and calculate metrics.
    saved = spark.read.parquet(str(staging))

    expressions = [
        F.count("*").alias("row_count")
    ]

    for metric, flag in FLAG_MAP.items():
        expressions.append(
            F.coalesce(
                F.sum(F.col(flag).cast("long")),
                F.lit(0)
            ).alias(metric)
        )

    actual = saved.agg(*expressions).first().asDict()
    expected = audit[source.name]

    if actual != expected:
        raise ValueError(
            f"Bronze validation failed: {source.name}\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )

    # Save checkpoint metadata inside staging.
    metadata = {
        "source_file": source.name,
        "metrics": actual,
    }

    marker = staging / "_bronze_complete.json"

    marker.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    # Publish only AFTER validation succeeds.
    if destination.exists():
        raise FileExistsError(destination)

    staging.rename(destination)

    print(
        f"PASSED | Rows: {actual['row_count']:,}",
        flush=True
    )


# ----------------------------------------
# 5. Run in manageable batches
# ----------------------------------------

try:

    selected = (
        pending_files
        if MAX_NEW_FILES is None
        else pending_files[:MAX_NEW_FILES]
    )

    for i in range(0, len(selected), BATCH_SIZE):

        batch = selected[i:i + BATCH_SIZE]

        print(
            f"\nBatch {i // BATCH_SIZE + 1} "
            f"({len(batch)} files)",
            flush=True
        )

        for source in batch:
            process_file(source)

    completed = sum(
        is_completed(p) for p in csv_files
    )

    print("\nBRONZE RUN FINISHED")
    print(f"Completed: {completed}/168")

    if completed == 168:
        total = sum(
            audit[p.name]["row_count"]
            for p in csv_files
        )

        print("ALL BRONZE FILES COMPLETED!")
        print(f"Validated raw rows: {total:,}")

finally:
    spark.stop()
