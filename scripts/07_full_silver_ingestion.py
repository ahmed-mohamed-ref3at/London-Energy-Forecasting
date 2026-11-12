
import os
import sys
import csv
import json
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark import StorageLevel


BASE = Path(__file__).resolve().parents[1]

BRONZE = BASE / "data/bronze/readings_by_file"
SILVER = BASE / "data/silver/readings_by_file"
STAGING = BASE / "data/silver/_staging"

AUDIT = BASE / "reports/raw_audit_batched.csv"
DUP_REPORTS = BASE / "reports/global_duplicate_buckets"
DST_REPORTS = BASE / "reports/dst_global_by_file"

SILVER.mkdir(parents=True, exist_ok=True)
STAGING.mkdir(parents=True, exist_ok=True)

# First test: 1 source file only
#MAX_NEW_FILES = 1
MAX_NEW_FILES = None
FLAG_MAP = {
    "missing_household_id": "is_missing_household_id",
    "missing_timestamp": "is_missing_timestamp",
    "invalid_timestamp": "is_invalid_timestamp",
    "missing_kwh": "is_missing_kwh",
    "invalid_kwh": "is_invalid_kwh",
    "negative_kwh": "is_negative_kwh",
}


# --------------------------------------
# 1. Verify completed global audits
# --------------------------------------

dup_files = [
    DUP_REPORTS / f"bucket_{i:02d}.json"
    for i in range(64)
]

if not all(p.is_file() for p in dup_files):
    raise ValueError("Global duplicate audit incomplete")

dup_results = [
    json.loads(p.read_text(encoding="utf-8"))
    for p in dup_files
]

if sum(x["conflicting_keys"] for x in dup_results) != 0:
    raise ValueError("Conflicting keys require review")

if sum(x["cross_file_keys"] for x in dup_results) != 0:
    raise ValueError("Cannot deduplicate per source")

EXPECTED_DUPLICATES = sum(
    x["extra_key_rows"] for x in dup_results
)

if EXPECTED_DUPLICATES != 115_453:
    raise ValueError("Unexpected global duplicate count")

dst_files = list(DST_REPORTS.glob("*.json"))

if len(dst_files) != 168:
    raise ValueError("Global DST audit incomplete")

dst_results = [
    json.loads(p.read_text(encoding="utf-8"))
    for p in dst_files
]

if sum(
    x["metrics"]["duplicate_keys"]
    for x in dst_results
) != 0:
    raise ValueError("DST duplicate keys require review")


# --------------------------------------
# 2. Load Raw Audit
# --------------------------------------

with AUDIT.open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    raw_audit = {
        row["source_file"]: {
            "row_count": int(row["row_count"]),
            **{
                name: int(row[name])
                for name in FLAG_MAP
            }
        }
        for row in csv.DictReader(file)
    }

if len(raw_audit) != 168:
    raise ValueError("Expected 168 Raw Audit records")

sources = sorted(
    [p for p in BRONZE.iterdir() if p.is_dir()],
    key=lambda p: int(p.name.rsplit("_", 1)[-1])
)

if len(sources) != 168:
    raise ValueError("Expected 168 Bronze folders")

if {p.name + ".csv" for p in sources} != set(raw_audit):
    raise ValueError("Bronze sources do not match audit")


# --------------------------------------
# 3. Checkpoint management
# --------------------------------------

def destination(source):
    return SILVER / source.name


def completed(source):
    folder = destination(source)

    if not folder.exists():
        return None

    marker = folder / "_silver_complete.json"

    if (
        not marker.is_file()
        or not (folder / "valid/_SUCCESS").is_file()
        or not (folder / "quarantine/_SUCCESS").is_file()
    ):
        raise RuntimeError(
            f"Incomplete Silver output: {folder}"
        )

    result = json.loads(
        marker.read_text(encoding="utf-8")
    )

    if (
        result.get("source_file") != source.name + ".csv"
        or result.get("bronze_rows") != raw_audit[
            source.name + ".csv"
        ]["row_count"]
    ):
        raise ValueError(
            f"Checkpoint mismatch: {source.name}"
        )

    return result


pending = [
    p for p in sources
    if completed(p) is None
]

selected = (
    pending if MAX_NEW_FILES is None
    else pending[:MAX_NEW_FILES]
)

print("=" * 55)
print("LONDON SMART METER - FULL SILVER INGESTION")
print("=" * 55)

print("Total source files:", len(sources))
print("Already completed:", 168 - len(pending))
print("Remaining:", len(pending))
print("Selected now:", len(selected))


# --------------------------------------
# 4. Start Spark
# --------------------------------------

spark = (
    SparkSession.builder
    .appName("LondonEnergyFullSilver")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.timestampType", "TIMESTAMP_NTZ")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# --------------------------------------
# 5. Process one source
# --------------------------------------

def process(source):

    name = source.name + ".csv"
    expected = raw_audit[name]

    final_dir = destination(source)
    temp_dir = STAGING / source.name

    if temp_dir.exists():
        raise RuntimeError(
            f"Unfinished staging directory: {temp_dir}"
        )

    print(f"\nProcessing: {source.name}", flush=True)

    bronze = spark.read.parquet(str(source))

    window = (
        Window
        .partitionBy("household_id", "timestamp_text")
        .orderBy(
            F.col("raw_household_id").asc_nulls_first(),
            F.col("raw_datetime").asc_nulls_first(),
            F.col("raw_tariff_group").asc_nulls_first(),
            F.col("raw_kwh").asc_nulls_first()
        )
    )

    classified = bronze.withColumn(
        "_duplicate_rank",
        F.row_number().over(window)
    )

    classified = classified.withColumn(
        "record_status",

        F.when(
            F.col("_duplicate_rank") > 1,
            "duplicate_key"
        )
        .when(
            F.col("is_missing_household_id"),
            "missing_household_id"
        )
        .when(
            F.col("is_missing_timestamp"),
            "missing_timestamp"
        )
        .when(
            F.col("is_invalid_timestamp")
            | F.col("reading_timestamp").isNull(),
            "invalid_timestamp"
        )
        .when(
            F.col("is_missing_kwh"),
            "missing_kwh"
        )
        .when(
            F.col("is_invalid_kwh")
            | F.col("kwh").isNull(),
            "invalid_kwh"
        )
        .when(
            F.col("is_negative_kwh"),
            "negative_kwh"
        )
        .otherwise("valid")
    )

    # Disk cache avoids recomputing the window
    # for each output, without relying on RAM.
    classified.persist(StorageLevel.DISK_ONLY)

    try:
        valid = (
            classified
            .filter(F.col("record_status") == "valid")
            .select(
                "source_file",
                "household_id",
                "tariff_group",
                "timestamp_text",
                "reading_timestamp",
                "kwh"
            )
        )

        quarantine = (
            classified
            .filter(F.col("record_status") != "valid")
            .drop("_duplicate_rank")
        )

        valid.write.mode("errorifexists").parquet(
            str(temp_dir / "valid")
        )

        quarantine.write.mode("errorifexists").parquet(
            str(temp_dir / "quarantine")
        )

    finally:
        classified.unpersist()


    # ----------------------------------
    # 6. Read-back validation
    # ----------------------------------

    saved_valid = spark.read.parquet(
        str(temp_dir / "valid")
    )

    saved_quarantine = spark.read.parquet(
        str(temp_dir / "quarantine")
    )

    valid_count = saved_valid.count()

    quality_expressions = [
        F.count("*").alias("quarantine_rows")
    ]

    for metric, flag in FLAG_MAP.items():
        quality_expressions.append(
            F.coalesce(
                F.sum(F.col(flag).cast("long")),
                F.lit(0)
            ).alias(metric)
        )

    q_metrics = (
        saved_quarantine
        .agg(*quality_expressions)
        .first()
        .asDict()
    )

    q_count = int(q_metrics["quarantine_rows"])

    for metric in FLAG_MAP:
        if int(q_metrics[metric]) != expected[metric]:
            raise ValueError(
                f"Quality flag mismatch: {name}, {metric}"
            )

    reasons = {
        row["record_status"]: row["count"]
        for row in (
            saved_quarantine
            .groupBy("record_status")
            .count()
            .collect()
        )
    }

    if valid_count + q_count != expected["row_count"]:
        raise ValueError(
            f"Row reconciliation failed: {name}"
        )

    duplicate_count = reasons.get("duplicate_key", 0)

    if source.name.endswith("_0") and duplicate_count != 688:
        raise ValueError(
            "First source duplicate count mismatch"
        )

    result = {
        "source_file": name,
        "bronze_rows": expected["row_count"],
        "valid_rows": valid_count,
        "quarantine_rows": q_count,
        "duplicate_rows": duplicate_count,
        "quality_flags": {
            metric: int(q_metrics[metric])
            for metric in FLAG_MAP
        },
        "quarantine_reasons": reasons,
    }

    marker = temp_dir / "_silver_complete.json"

    marker.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    # Publish both outputs together.
    if final_dir.exists():
        raise FileExistsError(final_dir)

    temp_dir.rename(final_dir)

    print(
        f"PASSED | Valid: {valid_count:,} "
        f"| Quarantine: {q_count:,}",
        flush=True
    )


# --------------------------------------
# 7. Execute and summarize
# --------------------------------------

try:
    for source in selected:
        process(source)

    results = [
        completed(source)
        for source in sources
    ]

    finished = [
        result for result in results
        if result is not None
    ]

    print("\nSILVER INGESTION PROGRESS")
    print(f"Completed: {len(finished)}/168")

    if len(finished) == 168:

        totals = {
            key: sum(r[key] for r in finished)
            for key in [
                "bronze_rows",
                "valid_rows",
                "quarantine_rows",
                "duplicate_rows"
            ]
        }

        for key, value in totals.items():
            print(f"{key}: {value:,}")

        if totals["bronze_rows"] != 167_932_474:
            raise ValueError("Unexpected Bronze total")

        if (
            totals["valid_rows"]
            + totals["quarantine_rows"]
            != totals["bronze_rows"]
        ):
            raise ValueError("Global reconciliation failed")

        if totals["duplicate_rows"] != EXPECTED_DUPLICATES:
            raise ValueError("Global duplicate mismatch")

        missing_total = sum(
            r["quality_flags"]["missing_kwh"]
            for r in finished
        )

        if missing_total != 5_560:
            raise ValueError("Global missing kWh mismatch")

        print("\nFULL SILVER VALIDATION: PASSED!")

    else:
        print("Partial results saved. Resume later.")

finally:
    spark.stop()
