"""Step 26A - resumable source-level preparation for TWO EDA Parquet datasets.

Run in local Windows conda environment from project root:
    python scripts/26_prepare_eda_hourly_fragments.py

FIRST RUN: MAX_NEW_FILES = 1. After success, change to None and rerun.

Silver VALID (168 sources) -> temporary, bucketed *household-hour fragments*.
These fragments preserve enough information for subsequent exact cross-source
combination into the approved 9-column `eda_hourly_history` and 13-column
`eda_household_daily` (Step 26B). Fragments are NOT final EDA datasets.

Why this intermediate is needed:
 A household can occur in more than one Silver source, and its two half-hour
 readings can fall in different source files. Computing 'complete homes' or
 complete days per source would produce WRONG results. We must combine the
 household-hour fragments across ALL source files first.

No timezone conversion, no imputation, no removal of valid consumption outliers.
TIMESTAMP_NTZ is treated as an original, naive wall-clock grid. Final span
coverage will be an observed-first/last estimate, not proof of meter operation.

All source files and 2013 Gold are read-only. Output/checkpoints can be safely
resumed; incomplete or inconsistent checkpoints cause an explicit failure.
"""

import json
import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession, functions as F
from pyspark.storagelevel import StorageLevel

BASE = Path(__file__).resolve().parents[1]
SILVER = BASE / "data" / "silver" / "readings_by_file"
WORK = BASE / "data" / "eda" / "_work_v1" / "source_hour_fragments"
CHECKPOINT_DIR = BASE / "reports" / "eda_build_v1" / "source_fragments"
HOUSEHOLD_SPANS = BASE / "reports" / "household_coverage_summary.csv"

# Pilot on one of the 168 Silver folders; then set to None and rerun.
#MAX_NEW_FILES = 1
MAX_NEW_FILES = None
EXPECTED_SOURCES = 168
EXPECTED_SILVER_VALID_ROWS = 167_811_461
BUCKETS = 32
VERSION = "eda_hour_fragments_v1_32_buckets"


def source_number(source: Path) -> int:
    return int(source.name.rsplit("_", 1)[-1])


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Leftover JSON temporary file; inspect: {temporary}")
    temporary.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_manifest(source: Path) -> dict:
    manifest = source / "_silver_complete.json"
    if not manifest.is_file() or not (source / "valid" / "_SUCCESS").is_file():
        raise RuntimeError(f"Step 15 Silver incomplete for {source.name}")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    if meta.get("source_file") != source.name + ".csv":
        raise ValueError(f"Unexpected source in Silver checkpoint: {source.name}")
    if type(meta.get("valid_rows")) is not int or meta["valid_rows"] <= 0:
        raise ValueError(f"Bad Silver valid count: {source.name}")
    return meta


def checked_done(source: Path, valid_count: int):
    target = WORK / source.name
    marker = CHECKPOINT_DIR / (source.name + ".json")
    if not target.exists() and not marker.exists():
        return None
    if (not target.is_dir() or not (target / "_SUCCESS").is_file()
            or not marker.is_file()):
        raise RuntimeError(
            f"Partial source output detected: {source.name}. "
            "Do not delete it automatically; inspect before retrying."
        )
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if (meta.get("version") != VERSION or meta.get("source") != source.name
            or meta.get("input_valid_rows") != valid_count
            or meta.get("combined_readings") != valid_count
            or meta.get("buckets") != BUCKETS):
        raise ValueError(f"Source checkpoint mismatch: {source.name}")
    written = sorted(p.name for p in target.glob("bucket=*") if p.is_dir())
    if written != meta.get("bucket_folders"):
        raise ValueError(f"Bucket folder list has changed: {source.name}")
    return meta


def make_spark():
    spark = (
        SparkSession.builder.appName("LondonEnergyEDAFragmentsV1")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.timestampType", "TIMESTAMP_NTZ")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def process_one(spark, source, valid_count):
    print(f"\nProcessing: {source.name}", flush=True)
    df = (
        spark.read.parquet(str(source / "valid"))
        .select("household_id", "reading_timestamp", "kwh", "tariff_group")
    )
    # Validate rather than filter: removing records here would invalidate Silver
    # reconciliation and silently alter EDA distributions/outliers.
    time_str = F.col("reading_timestamp").cast("string")
    bad_grid = (
        F.col("reading_timestamp").isNull()
        | (~F.minute("reading_timestamp").isin(0, 30))
        | (F.second("reading_timestamp") != 0)
        | (F.date_format("reading_timestamp", "SSSSSS") != "000000")
    )
    bad_value = (
        F.col("kwh").isNull()
        | F.isnan("kwh")
        | (F.abs(F.col("kwh")) == F.lit(float("inf")))
        | (F.col("kwh") < 0)
    )
    checks = df.agg(
        F.count("*").alias("rows"),
        F.sum(F.when(F.col("household_id").isNull() | bad_grid, 1).otherwise(0)).alias("bad_keys_or_grid"),
        F.sum(F.when(bad_value, 1).otherwise(0)).alias("bad_kwh"),
    ).first()
    if (int(checks["rows"]) != valid_count or checks["bad_keys_or_grid"] != 0
            or checks["bad_kwh"] != 0):
        raise ValueError(f"Invalid Silver source data: {source.name}: {checks.asDict()}")

    # A textual hour key avoids TIMESTAMP_NTZ/date_trunc type interactions seen
    # on this project's Windows Spark installation in Step 24.
    hh = (
        df.withColumn("hour_key", F.substring(time_str, 1, 13))
        .withColumn("slot", F.substring(time_str, 12, 2).cast("int") * F.lit(2)
                    + F.substring(time_str, 15, 2).cast("int") / F.lit(30))
        .withColumn("slot", F.col("slot").cast("int"))
        .withColumn("hour_int", F.substring(time_str, 12, 2).cast("int"))
        .withColumn("bucket", F.format_string("%02d", F.pmod(F.xxhash64("household_id"), F.lit(BUCKETS))))
    )
    peak_struct = F.struct(
        F.col("kwh").alias("kwh"),
        (-F.col("slot")).alias("negative_slot"),
    )
    fragments = (
        hh.groupBy("bucket", "household_id", "hour_key")
        .agg(
            F.count("*").alias("readings_observed"),
            F.sum("kwh").alias("energy_kwh_observed"),
            F.max(peak_struct).alias("peak"),
            F.sum(F.when(F.col("hour_int").between(17, 22), F.col("kwh"))
                  .otherwise(F.lit(0.0))).alias("evening_kwh"),
            F.sum(F.when(F.col("hour_int").between(0, 5), F.col("kwh"))
                  .otherwise(F.lit(0.0))).alias("overnight_kwh"),
            F.min("tariff_group").alias("tariff_min"),
            F.max("tariff_group").alias("tariff_max"),
        )
        .select(
            "bucket", "household_id", "hour_key", "readings_observed",
            "energy_kwh_observed",
            F.col("peak.kwh").alias("peak_halfhour_kwh"),
            (-F.col("peak.negative_slot")).cast("int").alias("peak_slot"),
            "evening_kwh", "overnight_kwh", "tariff_min", "tariff_max",
        )
    )

    target = WORK / source.name
    stage = WORK / ("_" + source.name + "_staging")
    marker = CHECKPOINT_DIR / (source.name + ".json")
    if target.exists() or stage.exists() or marker.exists():
        raise RuntimeError(f"Output already present; inspect: {target}, {stage}, {marker}")

    # Repartition small enough to run on a teammate's 16 GB laptop. On disk
    # 'bucket=XX' allows Step 26B to merge one bounded household hash group
    # at a time rather than joining all ~168 million readings at once.
    fragments.repartition(8, "bucket").write.partitionBy("bucket").mode("errorifexists").parquet(str(stage))

    staged = spark.read.parquet(str(stage))
    result = staged.agg(
        F.count("*").alias("hour_fragments"),
        F.sum("readings_observed").alias("combined_readings"),
        F.max("readings_observed").alias("max_readings_per_source_home_hour"),
        F.sum(F.when((F.col("readings_observed") <= 0)
                     | (F.col("readings_observed") > 2)
                     | (F.col("peak_slot") < 0)
                     | (F.col("peak_slot") > 47), 1).otherwise(0)).alias("invalid_groups"),
    ).first()
    if (int(result["combined_readings"]) != valid_count or result["invalid_groups"] != 0):
        raise ValueError(
            f"Fragment reconciliation FAILED for {source.name}: {result.asDict()}. "
            f"Staged files left for inspection: {stage}"
        )

    stage.rename(target)
    bucket_folders = sorted(p.name for p in target.glob("bucket=*") if p.is_dir())
    meta = {
        "version": VERSION,
        "source": source.name,
        "input_valid_rows": valid_count,
        "combined_readings": int(result["combined_readings"]),
        "hour_fragments": int(result["hour_fragments"]),
        "max_readings_per_source_home_hour": int(result["max_readings_per_source_home_hour"]),
        "buckets": BUCKETS,
        "bucket_folders": bucket_folders,
        "no_timezone_conversion": True,
        "outliers_retained": True,
        "is_intermediate_not_final_eda": True,
    }
    atomic_json(marker, meta)
    checked_done(source, valid_count)
    print(
        f"PASSED | Silver Valid: {valid_count:,} | "
        f"Household-hour fragments: {meta['hour_fragments']:,} | "
        f"Buckets: {len(bucket_folders)}", flush=True,
    )


def main():
    if not SILVER.is_dir() or not HOUSEHOLD_SPANS.is_file():
        raise FileNotFoundError("Silver or Step 17 household coverage report is missing")
    sources = sorted((p for p in SILVER.iterdir() if p.is_dir()), key=source_number)
    if len(sources) != EXPECTED_SOURCES:
        raise ValueError(f"Expected {EXPECTED_SOURCES} Silver folders, got {len(sources)}")
    sizes = {p.name: source_manifest(p)["valid_rows"] for p in sources}
    if sum(sizes.values()) != EXPECTED_SILVER_VALID_ROWS:
        raise ValueError("Silver total differs from Step 16 final validation")

    pending = [p for p in sources if checked_done(p, sizes[p.name]) is None]
    selected = pending if MAX_NEW_FILES is None else pending[:MAX_NEW_FILES]
    print("=" * 65)
    print("STEP 26A - PREPARE HISTORICAL EDA FRAGMENTS")
    print("=" * 65)
    print("Source: Silver Valid (all years, all 5,561 Silver households)")
    print("Total Silver sources:", len(sources))
    print("Already completed:", len(sources) - len(pending))
    print("Remaining:", len(pending))
    print("Selected now:", len(selected))
    print("Buckets:", BUCKETS)
    print("Final EDA schemas: 9 hourly columns + 13 daily columns (Step 26B)")
    print("Raw/Bronze/Silver/2013 Gold: READ-ONLY")

    spark = None
    try:
        if selected:
            spark = make_spark()
            for source in selected:
                process_one(spark, source, sizes[source.name])
    finally:
        if spark is not None:
            spark.stop()

    finished = sum(checked_done(p, sizes[p.name]) is not None for p in sources)
    print("\nEDA FRAGMENTS PROGRESS")
    print(f"Completed: {finished}/{EXPECTED_SOURCES}")
    if finished == EXPECTED_SOURCES:
        print("SOURCE FRAGMENTS: PASSED")
        print("Next: Step 26B merges the 32 buckets across all files, builds both approved EDA Parquets, and validates them.")
    else:
        print("Intermediate fragments saved. Rerun after setting MAX_NEW_FILES=None.")
    print("Intermediate storage:", WORK)
    print("Checkpoint folder:", CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
