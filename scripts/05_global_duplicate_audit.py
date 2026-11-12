
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

BUCKET_DIR = (
    BASE_DIR / "data" / "audit_workspace"
    / "duplicate_buckets" / "by_source"
)

RAW_AUDIT_PATH = (
    BASE_DIR / "reports" / "raw_audit_batched.csv"
)

REPORT_DIR = (
    BASE_DIR / "reports" / "global_duplicate_buckets"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

NUM_BUCKETS = 64

# First test: only one bucket
#MAX_NEW_BUCKETS = 1
MAX_NEW_BUCKETS = None


# ---------------------------------------
# 1. Validate source checkpoints
# ---------------------------------------

with RAW_AUDIT_PATH.open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    raw_audit = {
        row["source_file"]: int(row["row_count"])
        for row in csv.DictReader(file)
    }

if len(raw_audit) != 168:
    raise ValueError("Expected 168 Raw Audit records.")

sources = sorted(
    p for p in BUCKET_DIR.iterdir()
    if p.is_dir()
)

if len(sources) != 168:
    raise ValueError(
        f"Expected 168 prepared sources, found {len(sources)}"
    )

if {p.name + ".csv" for p in sources} != set(raw_audit):
    raise ValueError("Source folders do not match Raw Audit.")

for source in sources:

    marker = source / "_bucket_complete.json"

    if not marker.is_file():
        raise ValueError(
            f"Missing checkpoint: {source.name}"
        )

    if not (source / "_SUCCESS").is_file():
        raise ValueError(
            f"Missing Spark success marker: {source.name}"
        )

    metadata = json.loads(
        marker.read_text(encoding="utf-8")
    )

    if (
        metadata.get("source_file") != source.name + ".csv"
        or metadata.get("row_count") != raw_audit[
            source.name + ".csv"
        ]
        or metadata.get("num_buckets") != NUM_BUCKETS
    ):
        raise ValueError(
            f"Invalid source checkpoint: {source.name}"
        )

expected_total_rows = sum(raw_audit.values())


# ---------------------------------------
# 2. Per-bucket checkpoints
# ---------------------------------------

METRIC_NAMES = [
    "rows_in_bucket",
    "unique_keys",
    "duplicate_keys",
    "extra_key_rows",
    "conflicting_keys",
    "cross_file_keys",
    "extra_exact_rows",
]


def report_path(bucket_id):
    return REPORT_DIR / f"bucket_{bucket_id:02d}.json"


def read_completed(bucket_id):

    path = report_path(bucket_id)

    if not path.exists():
        return None

    result = json.loads(
        path.read_text(encoding="utf-8")
    )

    if (
        result.get("bucket_id") != bucket_id
        or result.get("num_buckets") != NUM_BUCKETS
        or result.get("source_count") != 168
        or any(
            name not in result
            for name in METRIC_NAMES
        )
    ):
        raise ValueError(
            f"Invalid bucket report: {path}"
        )

    return result


completed = {
    i: read_completed(i)
    for i in range(NUM_BUCKETS)
}

pending = [
    i for i in range(NUM_BUCKETS)
    if completed[i] is None
]

selected = (
    pending
    if MAX_NEW_BUCKETS is None
    else pending[:MAX_NEW_BUCKETS]
)

print("=" * 55)
print("LONDON SMART METER - GLOBAL DUPLICATE AUDIT")
print("=" * 55)

print("Total buckets:", NUM_BUCKETS)
print("Already completed:", NUM_BUCKETS - len(pending))
print("Remaining buckets:", len(pending))
print("Selected now:", len(selected))


# ---------------------------------------
# 3. Start Spark
# ---------------------------------------

spark = (
    SparkSession.builder
    .appName("LondonEnergyGlobalDuplicateAudit")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def safe_sum(expression):

    return F.coalesce(
        F.sum(expression),
        F.lit(0)
    )


# ---------------------------------------
# 4. Audit one bucket globally
# ---------------------------------------

def audit_bucket(bucket_id):

    print(
        f"\nProcessing bucket {bucket_id}/63",
        flush=True
    )

    # Find this bucket's Parquet parts
    # across ALL 168 source folders.
    parquet_files = []

    for source in sources:

        partition_folder = (
            source / f"audit_bucket={bucket_id}"
        )

        if partition_folder.is_dir():

            parquet_files.extend(
                str(p)
                for p in partition_folder.glob("*.parquet")
            )

    if not parquet_files:

        metrics = {
            name: 0 for name in METRIC_NAMES
        }

    else:

        # Read the actual Parquet part files,
        # not the parent source directories.
        df = (
            spark.read.parquet(*parquet_files)
            .select(
                "source_file",
                "household_id",
                "timestamp_text",
                "tariff_group",
                "kwh_text",
            )
        )

        # --------------------------------
        # A. Duplicate household/time keys
        # --------------------------------

        key_groups = (
            df.groupBy(
                "household_id",
                "timestamp_text"
            )
            .agg(
                F.count("*").alias("group_rows"),

                F.min("source_file").alias(
                    "first_source"
                ),

                F.max("source_file").alias(
                    "last_source"
                ),

                F.countDistinct(
                    F.struct(
                        "tariff_group",
                        "kwh_text"
                    )
                ).alias("value_variants")
            )
        )

        key_stats = key_groups.agg(

            safe_sum(
                F.col("group_rows")
            ).alias("rows_in_bucket"),

            F.count("*").alias("unique_keys"),

            safe_sum(
                F.when(
                    F.col("group_rows") > 1, 1
                ).otherwise(0)
            ).alias("duplicate_keys"),

            safe_sum(
                F.when(
                    F.col("group_rows") > 1,
                    F.col("group_rows") - 1
                ).otherwise(0)
            ).alias("extra_key_rows"),

            safe_sum(
                F.when(
                    F.col("value_variants") > 1, 1
                ).otherwise(0)
            ).alias("conflicting_keys"),

            safe_sum(
                F.when(
                    F.col("first_source")
                    != F.col("last_source"),
                    1
                ).otherwise(0)
            ).alias("cross_file_keys")

        ).first().asDict()

        # --------------------------------
        # B. Exact duplicate extra rows
        # --------------------------------

        exact_groups = (
            df.groupBy(
                "household_id",
                "timestamp_text",
                "tariff_group",
                "kwh_text"
            )
            .count()
        )

        exact_stats = exact_groups.agg(

            safe_sum(
                F.when(
                    F.col("count") > 1,
                    F.col("count") - 1
                ).otherwise(0)
            ).alias("extra_exact_rows")

        ).first()

        metrics = {
            **key_stats,
            "extra_exact_rows": exact_stats[
                "extra_exact_rows"
            ]
        }

    metrics = {
        name: int(metrics[name])
        for name in METRIC_NAMES
    }

    if (
        metrics["extra_exact_rows"]
        > metrics["extra_key_rows"]
    ):
        raise ValueError(
            f"Inconsistent duplicate counts: {bucket_id}"
        )

    # --------------------------------
    # C. Save validated checkpoint
    # --------------------------------

    result = {
        "bucket_id": bucket_id,
        "num_buckets": NUM_BUCKETS,
        "source_count": len(sources),
        **metrics
    }

    output = report_path(bucket_id)

    temporary = output.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    temporary.replace(output)

    print(
        f"PASSED | Rows: {metrics['rows_in_bucket']:,}",
        flush=True
    )

    print(
        "Extra duplicate rows:",
        f"{metrics['extra_key_rows']:,}",
        flush=True
    )


# ---------------------------------------
# 5. Run and summarize
# ---------------------------------------

try:

    for bucket_id in selected:
        audit_bucket(bucket_id)

    reports = [
        read_completed(i)
        for i in range(NUM_BUCKETS)
    ]

    finished = [
        report for report in reports
        if report is not None
    ]

    print("\nGLOBAL DUPLICATE AUDIT PROGRESS")
    print(f"Completed: {len(finished)}/64")

    if len(finished) == NUM_BUCKETS:

        totals = {
            name: sum(
                report[name] for report in finished
            )
            for name in METRIC_NAMES
        }

        if totals["rows_in_bucket"] != expected_total_rows:
            raise ValueError(
                "GLOBAL ROW COUNT MISMATCH!\n"
                f"Expected: {expected_total_rows}\n"
                f"Actual: {totals['rows_in_bucket']}"
            )

        print("\nFINAL GLOBAL RESULTS")

        for name, value in totals.items():
            print(f"{name}: {value:,}")

        print("\nGLOBAL ROW RECONCILIATION: PASSED!")

    else:
        print("Partial results saved. Resume later.")

finally:
    spark.stop()
