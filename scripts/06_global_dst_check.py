
import os
import sys
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

REPORT_DIR = (
    BASE_DIR / "reports" / "dst_global_by_file"
)

DUPLICATE_REPORT_DIR = (
    BASE_DIR / "reports" / "global_duplicate_buckets"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

FALLBACK_DATES = ["2012-10-28", "2013-10-27"]

# Test 5 files first
#MAX_NEW_FILES = 5
MAX_NEW_FILES = None
# ----------------------------------
# 1. Verify global duplicate audit
# ----------------------------------

duplicate_reports = [
    DUPLICATE_REPORT_DIR / f"bucket_{i:02d}.json"
    for i in range(64)
]

if not all(p.is_file() for p in duplicate_reports):
    raise ValueError(
        "Global duplicate audit must be completed first."
    )

cross_file_keys = sum(
    json.loads(p.read_text(encoding="utf-8"))[
        "cross_file_keys"
    ]
    for p in duplicate_reports
)

if cross_file_keys != 0:
    raise ValueError(
        "Cross-file duplicates exist. "
        "Per-file DST aggregation is insufficient."
    )


# ----------------------------------
# 2. Discover source files
# ----------------------------------

sources = sorted(
    [
        p for p in BRONZE_DIR.iterdir()
        if p.is_dir()
    ],
    key=lambda p: int(p.name.rsplit("_", 1)[-1])
)

if len(sources) != 168:
    raise ValueError("Expected 168 Bronze folders")


def report_path(source):
    return REPORT_DIR / f"{source.name}.json"


def read_completed(source):
    path = report_path(source)

    if not path.exists():
        return None

    result = json.loads(
        path.read_text(encoding="utf-8")
    )

    if (
        result.get("source_file") != source.name
        or result.get("fallback_dates") != FALLBACK_DATES
        or result.get("checked_hour") != 1
    ):
        raise ValueError(
            f"Invalid DST checkpoint: {source.name}"
        )

    return result


pending = [
    source for source in sources
    if read_completed(source) is None
]

selected = (
    pending
    if MAX_NEW_FILES is None
    else pending[:MAX_NEW_FILES]
)

print("=" * 55)
print("LONDON SMART METER - GLOBAL DST CHECK")
print("=" * 55)

print("Total files:", len(sources))
print("Already completed:", 168 - len(pending))
print("Remaining:", len(pending))
print("Selected now:", len(selected))


# ----------------------------------
# 3. Start Spark
# ----------------------------------

spark = (
    SparkSession.builder
    .appName("LondonEnergyGlobalDSTCheck")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.timestampType", "TIMESTAMP_NTZ")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def safe_sum(expression):
    return F.coalesce(
        F.sum(expression),
        F.lit(0)
    )


# ----------------------------------
# 4. Process each Bronze file
# ----------------------------------

try:

    for source in selected:

        print(
            f"\nProcessing: {source.name}",
            flush=True
        )

        df = spark.read.parquet(str(source)).select(
            "household_id",
            "reading_timestamp",
            "tariff_group",
            "kwh_text"
        )

        dst = df.filter(
            F.to_date("reading_timestamp")
            .cast("string")
            .isin(*FALLBACK_DATES)
            &
            (F.hour("reading_timestamp") == 1)
        )

        grouped = (
            dst.groupBy(
                "household_id",
                "reading_timestamp"
            )
            .agg(
                F.count("*").alias("group_rows"),

                F.countDistinct(
                    F.struct(
                        "tariff_group",
                        "kwh_text"
                    )
                ).alias("value_variants")
            )
        )

        stats = grouped.agg(

            F.count("*").alias("unique_keys"),

            safe_sum(
                F.col("group_rows")
            ).alias("rows_in_window"),

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
            ).alias("extra_rows"),

            safe_sum(
                F.when(
                    F.col("value_variants") > 1, 1
                ).otherwise(0)
            ).alias("conflicting_keys")

        ).first().asDict()

        metrics = {
            name: int(value)
            for name, value in stats.items()
        }

        result = {
            "source_file": source.name,
            "fallback_dates": FALLBACK_DATES,
            "checked_hour": 1,
            "metrics": metrics
        }

        output = report_path(source)
        temporary = output.with_suffix(".tmp")

        temporary.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8"
        )

        temporary.replace(output)

        print(
            "PASSED | Rows in DST windows:",
            metrics["rows_in_window"],
            flush=True
        )

        print(
            "Duplicate keys:",
            metrics["duplicate_keys"],
            flush=True
        )

    # ----------------------------------
    # 5. Summarize saved checkpoints
    # ----------------------------------

    completed = [
        read_completed(source)
        for source in sources
    ]

    completed = [
        item for item in completed
        if item is not None
    ]

    print("\nGLOBAL DST CHECK PROGRESS")
    print(f"Completed: {len(completed)}/168")

    if len(completed) == 168:

        metric_names = [
            "unique_keys",
            "rows_in_window",
            "duplicate_keys",
            "extra_rows",
            "conflicting_keys",
        ]

        totals = {
            name: sum(
                item["metrics"][name]
                for item in completed
            )
            for name in metric_names
        }

        print("\nFINAL GLOBAL DST RESULTS")

        for name, value in totals.items():
            print(f"{name}: {value:,}")

        print("\nGLOBAL DST CHECK COMPLETED!")

    else:
        print("Partial results saved. Resume later.")

finally:
    spark.stop()
