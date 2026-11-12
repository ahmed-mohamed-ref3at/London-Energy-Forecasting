
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

BRONZE_DIR = (
    BASE_DIR / "data" / "bronze" / "readings_by_file"
)

WORK_DIR = (
    BASE_DIR / "data" / "audit_workspace"
    / "duplicate_buckets"
)

OUTPUT_DIR = WORK_DIR / "by_source"
STAGING_DIR = WORK_DIR / "_staging"

AUDIT_PATH = (
    BASE_DIR / "reports" / "raw_audit_batched.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGING_DIR.mkdir(parents=True, exist_ok=True)


# Initial test: 2 files only
#MAX_NEW_FILES = 2
MAX_NEW_FILES = None

NUM_BUCKETS = 64


# --------------------------------------
# 1. Load expected source row counts
# --------------------------------------

with AUDIT_PATH.open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    raw_audit = {
        row["source_file"]: int(row["row_count"])
        for row in csv.DictReader(file)
    }

if len(raw_audit) != 168:
    raise ValueError("Expected 168 audit records")


sources = sorted(
    BRONZE_DIR.iterdir(),
    key=lambda p: int(p.name.rsplit("_", 1)[-1])
)

sources = [p for p in sources if p.is_dir()]

if len(sources) != 168:
    raise ValueError("Expected 168 Bronze source folders")


def completed(source):
    destination = OUTPUT_DIR / source.name

    if not destination.exists():
        return False

    marker = destination / "_bucket_complete.json"

    if not marker.is_file():
        raise RuntimeError(
            f"Incomplete output: {destination}"
        )

    metadata = json.loads(
        marker.read_text(encoding="utf-8")
    )

    expected = raw_audit[source.name + ".csv"]

    if (
        metadata.get("source_file") != source.name + ".csv"
        or metadata.get("row_count") != expected
        or metadata.get("num_buckets") != NUM_BUCKETS
        or not (destination / "_SUCCESS").is_file()
    ):
        raise ValueError(
            f"Invalid checkpoint: {source.name}"
        )

    return True


pending = [p for p in sources if not completed(p)]

selected = (
    pending if MAX_NEW_FILES is None
    else pending[:MAX_NEW_FILES]
)

print("=" * 55)
print("PREPARE GLOBAL DUPLICATE AUDIT BUCKETS")
print("=" * 55)

print("Completed sources:", 168 - len(pending))
print("Remaining sources:", len(pending))
print("Selected now:", len(selected))
print("Number of buckets:", NUM_BUCKETS)


# --------------------------------------
# 2. Start Spark
# --------------------------------------

spark = (
    SparkSession.builder
    .appName("LondonEnergyDuplicateBuckets")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


try:
    for source in selected:

        destination = OUTPUT_DIR / source.name
        staging = STAGING_DIR / source.name

        if staging.exists():
            raise RuntimeError(
                f"Unfinished staging directory: {staging}"
            )

        print(
            f"\nProcessing: {source.name}",
            flush=True
        )

        df = spark.read.parquet(str(source)).select(
            "source_file",
            "household_id",
            "timestamp_text",
            "tariff_group",
            "kwh_text",
        )

        # Same household -> same bucket
        df = df.withColumn(
            "audit_bucket",
            F.pmod(
                F.xxhash64(
                    F.coalesce(
                        F.col("household_id"),
                        F.lit("__NULL_HOUSEHOLD__")
                    )
                ),
                F.lit(NUM_BUCKETS)
            ).cast("int")
        )

        # Save data partitioned by hash bucket
        (
            df.repartition(4, "audit_bucket")
            .write
            .mode("errorifexists")
            .partitionBy("audit_bucket")
            .parquet(str(staging))
        )

        # Validate all rows were preserved
        saved_count = spark.read.parquet(
            str(staging)
        ).count()

        expected_count = raw_audit[
            source.name + ".csv"
        ]

        if saved_count != expected_count:
            raise ValueError(
                f"Row mismatch: {source.name}\n"
                f"Expected: {expected_count}\n"
                f"Actual: {saved_count}"
            )

        marker_data = {
            "source_file": source.name + ".csv",
            "row_count": saved_count,
            "num_buckets": NUM_BUCKETS,
        }

        (staging / "_bucket_complete.json").write_text(
            json.dumps(marker_data, indent=2),
            encoding="utf-8"
        )

        # Publish only after validation
        staging.rename(destination)

        print(
            f"PASSED | Rows: {saved_count:,}",
            flush=True
        )

    total_completed = sum(
        completed(p) for p in sources
    )

    print("\nBUCKET PREPARATION FINISHED")
    print(f"Completed: {total_completed}/168")

finally:
    spark.stop()
