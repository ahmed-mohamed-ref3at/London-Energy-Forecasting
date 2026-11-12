
import os
import sys
import csv
from pathlib import Path

# Ensure Spark uses our Conda Python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ---------------------------------------
# 1. Project paths
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw" / "csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "raw_audit_batched.csv"


# ---------------------------------------
# 2. Execution settings
# ---------------------------------------

BATCH_SIZE = 5

# Test only one batch for now.
# Later change to None for all batches.

#MAX_BATCHES = 1
MAX_BATCHES = None

csv_files = sorted(RAW_DIR.rglob("*.csv"))

if len(csv_files) != 168:
    raise ValueError(
        f"Expected 168 CSV files, found {len(csv_files)}"
    )

if len({p.name for p in csv_files}) != len(csv_files):
    raise ValueError("Duplicate CSV filenames found!")


# ---------------------------------------
# 3. Report columns
# ---------------------------------------

REPORT_COLUMNS = [
    "source_file",
    "row_count",
    "unique_households_in_file",
    "missing_household_id",
    "missing_timestamp",
    "invalid_timestamp",
    "missing_kwh",
    "invalid_kwh",
    "negative_kwh",
    "min_timestamp",
    "max_timestamp",
]


# ---------------------------------------
# 4. Load previous progress
# ---------------------------------------

completed_files = set()

if REPORT_PATH.exists():

    with REPORT_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != REPORT_COLUMNS:
            raise ValueError(
                "Existing audit report has different columns."
            )

        for row in reader:
            completed_files.add(row["source_file"])


remaining_files = [
    p for p in csv_files
    if p.name not in completed_files
]

print("=" * 55)
print("LONDON SMART METER - BATCHED RAW AUDIT")
print("=" * 55)

print("Total CSV files:", len(csv_files))
print("Already completed:", len(completed_files))
print("Remaining:", len(remaining_files))
print("Batch size:", BATCH_SIZE)


# ---------------------------------------
# 5. Start Spark
# ---------------------------------------

spark = (
    SparkSession.builder
    .appName("LondonEnergyBatchedAudit")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def is_missing(column_name):

    col = F.col(column_name)

    return (
        col.isNull()
        | (col == "")
        | (F.lower(col) == "null")
    )


def count_if(condition):

    return F.sum(
        F.when(condition, 1).otherwise(0)
    )


try:

    batches = [
        remaining_files[i:i + BATCH_SIZE]
        for i in range(0, len(remaining_files), BATCH_SIZE)
    ]

    for batch_number, batch_files in enumerate(
        batches, start=1
    ):

        if (
            MAX_BATCHES is not None
            and batch_number > MAX_BATCHES
        ):
            break

        print(
            f"\nProcessing batch {batch_number} "
            f"({len(batch_files)} CSV files)"
        )

        # -----------------------------------
        # 6. Read current batch
        # -----------------------------------

        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .option("mode", "PERMISSIVE")
            .csv([str(p) for p in batch_files])
        )

        df = df.toDF(
            *[col.strip() for col in df.columns]
        )

        # -----------------------------------
        # 7. Standardize data
        # -----------------------------------

        df = df.select(

            F.regexp_extract(
                F.input_file_name(),
                r"([^/]+\.csv)$",
                1
            ).alias("source_file"),

            F.trim(
                F.col("LCLid")
            ).alias("household_id"),

            F.trim(
                F.col("DateTime")
            ).alias("timestamp_text"),

            F.trim(
                F.col("KWH/hh (per half hour)")
            ).alias("kwh_text"),
        )

        df = df.withColumn(
            "timestamp",
            F.try_to_timestamp(
                F.col("timestamp_text"),
                F.lit("yyyy-MM-dd HH:mm:ss.SSSSSSS")
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

        # -----------------------------------
        # 8. Calculate audit metrics
        # -----------------------------------

        audit = (
            df.groupBy("source_file")
            .agg(

                F.count("*").alias("row_count"),

                F.countDistinct(
                    "household_id"
                ).alias("unique_households_in_file"),

                count_if(
                    is_missing("household_id")
                ).alias("missing_household_id"),

                count_if(
                    is_missing("timestamp_text")
                ).alias("missing_timestamp"),

                count_if(
                    ~is_missing("timestamp_text")
                    & F.col("timestamp").isNull()
                ).alias("invalid_timestamp"),

                count_if(
                    is_missing("kwh_text")
                ).alias("missing_kwh"),

                count_if(
                    ~is_missing("kwh_text")
                    & F.col("kwh").isNull()
                ).alias("invalid_kwh"),

                count_if(
                    F.col("kwh") < 0
                ).alias("negative_kwh"),

                F.min("timestamp").alias(
                    "min_timestamp"
                ),

                F.max("timestamp").alias(
                    "max_timestamp"
                ),
            )
        )

        # Collect only the small batch report
        results = audit.collect()

        # Verify that all files were included
        expected_names = {p.name for p in batch_files}
        actual_names = {
            row["source_file"] for row in results
        }

        if expected_names != actual_names:
            raise ValueError(
                "Audit filenames do not match batch files!"
            )

        # -----------------------------------
        # 9. Save progress immediately
        # -----------------------------------

        new_report = (
            not REPORT_PATH.exists()
            or REPORT_PATH.stat().st_size == 0
        )

        with REPORT_PATH.open(
            "a", encoding="utf-8", newline=""
        ) as file:

            writer = csv.writer(file)

            if new_report:
                writer.writerow(REPORT_COLUMNS)

            for row in results:
                writer.writerow([
                    row[column]
                    for column in REPORT_COLUMNS
                ])

        completed_files.update(actual_names)

        batch_rows = sum(
            row["row_count"] for row in results
        )

        print("Batch completed successfully!")
        print("Rows in this batch:", batch_rows)
        print(
            "Files completed overall:",
            f"{len(completed_files)}/168"
        )
        print("Progress saved:", REPORT_PATH)

    print("\nAudit run finished.")

    if len(completed_files) == 168:
        print("All 168 CSV files audited!")
    else:
        print(
            "Resume later to process the remaining files."
        )

finally:
    spark.stop()
