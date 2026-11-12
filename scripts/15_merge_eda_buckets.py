"""Step 26B: Build exactly TWO EDA Parquet datasets from Step 26A fragments.

Usage (from project root; conda env london-energy):
    python scripts/26_merge_eda_buckets.py

1) Run with MAX_NEW_BUCKETS=1, inspect pilot output.
2) Change MAX_NEW_BUCKETS=None and rerun. The 31 remaining buckets are
   processed once each; the final two EDA datasets publish only after 32/32.

Inputs:   data/eda/_work_v1/source_hour_fragments/<source>/bucket=XX/
          reports/eda_build_v1/source_fragments/<source>.json
          reports/household_coverage_summary.csv
Outputs:  data/eda/eda_hourly_history/           (exactly 9 columns)
          data/eda/eda_household_daily/           (exactly 13 columns)
          reports/eda_build_v1/final_validation.json

Intermediates remain in data/eda/_work_v1; do not treat them as final EDA
outputs. No changes are made to Silver, Bronze, Raw or the 2013 Gold.

Coverage uses each home's inclusive first-to-last VALID reading span on the
original TIMESTAMP_NTZ half-hour grid. It is an *observed-span estimate*,
not proof that an installed meter was operating on every intervening day.
Incomplete days have NULL complete-day consumption. No filling, clipping,
exclusion of outliers or time-zone transformation is applied.
"""

import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.storagelevel import StorageLevel

BASE = Path(__file__).resolve().parents[1]
WORK = BASE / "data" / "eda" / "_work_v1"
FRAGMENTS = WORK / "source_hour_fragments"
MERGED_WORK = WORK / "merged_by_bucket"
EDA_ROOT = BASE / "data" / "eda"
FINAL_HOURLY = EDA_ROOT / "eda_hourly_history"
FINAL_DAILY = EDA_ROOT / "eda_household_daily"
REPORTS = BASE / "reports" / "eda_build_v1"
SOURCE_MARKERS = REPORTS / "source_fragments"
BUCKET_MARKERS = REPORTS / "merged_buckets"
GLOBAL_MARKER = REPORTS / "final_validation.json"
SPANS_CSV = BASE / "reports" / "household_coverage_summary.csv"

MAX_NEW_BUCKETS = None     # Pilot 1 bucket. After PASSED, set to None.
BUCKETS = 32
EXPECTED_SOURCES = 168
EXPECTED_SILVER_READINGS = 167_811_461
EXPECTED_SILVER_HOUSEHOLDS = 5_561
EXPECTED_2013_SILVER_READINGS = 93_024_217
VERSION = "eda_final_v1_from_hour_fragments_32_buckets"
SOURCE_VERSION = "eda_hour_fragments_v1_32_buckets"

HOURLY_COLUMNS = [
    "hour_start", "energy_kwh_observed", "energy_kwh_complete_homes",
    "homes_reporting", "homes_complete", "readings_observed",
    "readings_expected_in_span", "coverage_pct_in_span",
    "avg_kwh_per_complete_home",
]
DAILY_COLUMNS = [
    "date", "household_id", "tariff_group", "energy_kwh_observed",
    "energy_kwh_complete_day", "readings_observed",
    "readings_expected_in_span", "coverage_pct_in_span",
    "is_complete_day", "peak_halfhour_kwh", "peak_hour",
    "evening_energy_share_pct", "overnight_energy_share_pct",
]


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        raise RuntimeError(f"Leftover marker temporary file: {tmp}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def make_spark():
    spark = (
        SparkSession.builder.appName("LondonEnergyHistoricalEDAFinalV1")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.timestampType", "TIMESTAMP_NTZ")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def get_source_paths():
    if not FRAGMENTS.is_dir() or not SOURCE_MARKERS.is_dir():
        raise FileNotFoundError("Run Step 26A first; its fragments/checkpoints are missing")
    sources = sorted(
        [p for p in FRAGMENTS.iterdir() if p.is_dir() and p.name.startswith("LCL-June2015v2_")],
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    if len(sources) != EXPECTED_SOURCES:
        raise ValueError(f"Expected 168 fragment sources, got {len(sources)}")
    expected_names = [f"LCL-June2015v2_{i}" for i in range(EXPECTED_SOURCES)]
    if [p.name for p in sources] != expected_names:
        raise ValueError("Fragment source numbering has missing or unexpected sources")
    total = 0
    source_buckets = {}
    for source in sources:
        marker_path = SOURCE_MARKERS / (source.name + ".json")
        if not marker_path.is_file() or not (source / "_SUCCESS").is_file():
            raise RuntimeError(f"Incomplete Step 26A output: {source.name}")
        meta = json.loads(marker_path.read_text(encoding="utf-8"))
        if (meta.get("version") != SOURCE_VERSION or meta.get("source") != source.name
                or meta.get("buckets") != BUCKETS
                or meta.get("combined_readings") != meta.get("input_valid_rows")):
            raise ValueError(f"Inconsistent Step 26A marker: {source.name}")
        present = sorted(p.name for p in source.glob("bucket=*") if p.is_dir())
        if present != meta.get("bucket_folders"):
            raise ValueError(f"Step 26A bucket folders changed: {source.name}")
        for bucket in present:
            n = int(bucket.split("=")[1])
            if not (0 <= n < BUCKETS):
                raise ValueError(f"Out-of-range bucket in {source.name}: {bucket}")
            source_buckets.setdefault(n, []).append(source / bucket)
        total += meta["combined_readings"]
    if total != EXPECTED_SILVER_READINGS:
        raise ValueError(f"Fragment checkpoint total {total} != Silver {EXPECTED_SILVER_READINGS}")
    return source_buckets


def parse_spans():
    if not SPANS_CSV.is_file():
        raise FileNotFoundError(f"Step 17 household span report missing: {SPANS_CSV}")
    result = []
    half_hour = timedelta(minutes=30)
    expected_total = 0
    with SPANS_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        needed = {"household_id", "first_timestamp", "last_timestamp", "observed_readings"}
        if not needed.issubset(set(r.fieldnames or [])):
            raise ValueError("Step 17 household coverage CSV is missing expected columns")
        for x in r:
            first = datetime.fromisoformat(x["first_timestamp"])
            last = datetime.fromisoformat(x["last_timestamp"])
            if (first > last or first.minute not in (0, 30)
                    or last.minute not in (0, 30)
                    or any((first.second, last.second, first.microsecond, last.microsecond))):
                raise ValueError(f"Off-grid or invalid first/last span: {x['household_id']}")
            expected = int((last - first) / half_hour) + 1
            if expected < int(x["observed_readings"]):
                raise ValueError(f"Observed readings exceed span expectation: {x['household_id']}")
            result.append((x["household_id"], x["first_timestamp"], x["last_timestamp"],
                           first.hour * 2 + first.minute // 30,
                           last.hour * 2 + last.minute // 30))
            expected_total += expected
    if len(result) != EXPECTED_SILVER_HOUSEHOLDS:
        raise ValueError(f"Expected {EXPECTED_SILVER_HOUSEHOLDS} households, got {len(result)}")
    if len({r[0] for r in result}) != len(result):
        raise ValueError("Duplicate household IDs in Step 17 coverage summary")
    return result, expected_total


def hour_expected_grid(spans):
    """O(households + hours) exact observed-span half-hour slots per hour.

    A full covered hour contributes two slots; start-at-:30 and end-at-:00
    each remove one slot from their corresponding boundary hour.
    """
    start = min(datetime.fromisoformat(x[1]).replace(minute=0) for x in spans)
    end = max(datetime.fromisoformat(x[2]).replace(minute=0) for x in spans)
    n = int((end - start) / timedelta(hours=1)) + 1
    diff = [0] * (n + 1)
    corrections = [0] * n
    for _, start_str, end_str, _, _ in spans:
        a = datetime.fromisoformat(start_str)
        b = datetime.fromisoformat(end_str)
        ai = int((a.replace(minute=0) - start) / timedelta(hours=1))
        bi = int((b.replace(minute=0) - start) / timedelta(hours=1))
        diff[ai] += 2
        diff[bi + 1] -= 2
        if a.minute == 30:
            corrections[ai] -= 1
        if b.minute == 0:
            corrections[bi] -= 1
    expected = 0
    rows = []
    for i in range(n):
        expected += diff[i]
        value = expected + corrections[i]
        if value < 0 or value > 2 * len(spans):
            raise ValueError(f"Invalid hourly span grid at index {i}: {value}")
        ts = start + timedelta(hours=i)
        rows.append((ts.strftime("%Y-%m-%d %H"), int(value)))
    return rows


def bucket_dir(bucket):
    return MERGED_WORK / f"bucket_{bucket:02d}"


def bucket_state(bucket):
    root = bucket_dir(bucket)
    marker = BUCKET_MARKERS / f"bucket_{bucket:02d}.json"
    hour = root / "hourly"
    day = root / "daily"
    if not any(p.exists() for p in (root, hour, day, marker)):
        return None
    if (not marker.is_file() or not (hour / "_SUCCESS").is_file()
            or not (day / "_SUCCESS").is_file()):
        raise RuntimeError(f"Partial bucket {bucket:02d} detected; inspect {root} and {marker}")
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if meta.get("version") != VERSION or meta.get("bucket") != bucket:
        raise ValueError(f"Invalid bucket checkpoint: {bucket:02d}")
    return meta


def present_bucket_fragments(source_buckets, bucket):
    paths = source_buckets.get(bucket, [])
    if not paths:
        raise ValueError(f"Bucket {bucket:02d} has no fragments")
    return [str(p) for p in paths]


def process_bucket(spark, bucket, source_buckets, span_df):
    print(f"\nProcessing bucket {bucket:02d}/31 ...", flush=True)
    root = bucket_dir(bucket)
    hour_output = root / "hourly"
    daily_output = root / "daily"
    root.mkdir(parents=True, exist_ok=False)
    raw = spark.read.parquet(*present_bucket_fragments(source_buckets, bucket))
    # All readings that belong to one home/hour are in ONE hash bucket.
    home_hour = (
        raw.groupBy("household_id", "hour_key")
        .agg(
            F.sum("readings_observed").cast("long").alias("n"),
            F.sum("energy_kwh_observed").alias("energy"),
            F.max(F.struct(
                F.col("peak_halfhour_kwh").alias("kwh"),
                (-F.col("peak_slot")).alias("neg_slot")
            )).alias("peak"),
            F.sum("evening_kwh").alias("evening_kwh"),
            F.sum("overnight_kwh").alias("overnight_kwh"),
            F.min("tariff_min").alias("tariff_min"),
            F.max("tariff_max").alias("tariff_max"),
        )
        .withColumn("peak_halfhour_kwh", F.col("peak.kwh"))
        .withColumn("peak_slot", (-F.col("peak.neg_slot")).cast("int"))
        .drop("peak")
        .persist(StorageLevel.DISK_ONLY)
    )
    hh_stats = home_hour.agg(
        F.count("*").alias("home_hours"),
        F.sum("n").alias("readings"),
        F.sum(F.when((F.col("n") < 1) | (F.col("n") > 2)
                     | (F.col("peak_slot") < 0) | (F.col("peak_slot") > 47), 1)
              .otherwise(0)).alias("bad_home_hours"),
        F.sum("energy").alias("observed_kwh"),
    ).first()
    if int(hh_stats["bad_home_hours"]) != 0:
        raise ValueError(f"Bucket {bucket:02d}: >2 readings per home/hour or bad peak slot")

    hour_part = (
        home_hour.groupBy("hour_key")
        .agg(
            F.sum("energy").alias("energy_kwh_observed"),
            F.sum(F.when(F.col("n") == 2, F.col("energy")).otherwise(0.0))
             .alias("energy_kwh_complete_homes"),
            F.count("*").cast("long").alias("homes_reporting"),
            F.sum(F.when(F.col("n") == 2, 1).otherwise(0)).cast("long").alias("homes_complete"),
            F.sum("n").cast("long").alias("readings_observed"),
        )
    )
    hour_part.coalesce(2).write.mode("errorifexists").parquet(str(hour_output))

    # Compute daily rows AFTER cross-file home/hour merge. Zero-reading days
    # inside the observed span are inserted by a per-household calendar grid.
    day_observed = (
        home_hour.withColumn("date", F.to_date(F.substring("hour_key", 1, 10)))
        .groupBy("household_id", "date")
        .agg(
            F.sum("n").cast("long").alias("readings_observed"),
            F.sum("energy").alias("energy_kwh_observed"),
            F.sum("evening_kwh").alias("evening_kwh"),
            F.sum("overnight_kwh").alias("overnight_kwh"),
            F.max(F.struct(
                F.col("peak_halfhour_kwh").alias("kwh"),
                (-F.col("peak_slot")).alias("neg_slot"),
            )).alias("day_peak"),
            F.min("tariff_min").alias("day_tariff_min"),
            F.max("tariff_max").alias("day_tariff_max"),
        )
    )
    home_tariff = home_hour.groupBy("household_id").agg(
        F.min("tariff_min").alias("home_tariff_min"),
        F.max("tariff_max").alias("home_tariff_max"),
    )

    spans = (
        span_df.where(F.col("bucket") == bucket)
        .drop("bucket")
        .withColumn("first_day", F.to_date(F.substring("first_timestamp", 1, 10)))
        .withColumn("last_day", F.to_date(F.substring("last_timestamp", 1, 10)))
    )
    days = (
        spans.withColumn("date", F.explode(F.sequence(
            F.col("first_day"), F.col("last_day"), F.expr("INTERVAL 1 DAY")
        )))
        .withColumn("first_covered_slot", F.when(
            F.col("date") == F.col("first_day"), F.col("first_slot")
        ).otherwise(F.lit(0)))
        .withColumn("last_covered_slot", F.when(
            F.col("date") == F.col("last_day"), F.col("last_slot")
        ).otherwise(F.lit(47)))
        .withColumn("readings_expected_in_span",
                    (F.col("last_covered_slot") - F.col("first_covered_slot") + 1).cast("long"))
        .select("household_id", "date", "readings_expected_in_span")
    )
    daily = (
        days.join(day_observed, ["household_id", "date"], "left")
        .join(home_tariff, "household_id", "left")
        .withColumn("readings_observed",
                    F.coalesce(F.col("readings_observed"), F.lit(0)).cast("long"))
        .withColumn("is_complete_day",
                    (F.col("readings_expected_in_span") == 48)
                    & (F.col("readings_observed") == 48))
        .withColumn("coverage_pct_in_span", F.round(
            100.0 * F.col("readings_observed") / F.col("readings_expected_in_span"), 4
        ))
        .withColumn("energy_kwh_complete_day", F.when(
            F.col("is_complete_day"), F.col("energy_kwh_observed")
        ).otherwise(F.lit(None).cast("double")))
        .withColumn("tar_min", F.coalesce("day_tariff_min", "home_tariff_min"))
        .withColumn("tar_max", F.coalesce("day_tariff_max", "home_tariff_max"))
        .withColumn("tariff_group", F.when(
            F.col("tar_min").isNull(), F.lit(None).cast("string")
        ).when(F.col("tar_min") == F.col("tar_max"), F.col("tar_min"))
          .otherwise(F.lit("MIXED")))
        .select(
            "date", "household_id", "tariff_group", "energy_kwh_observed",
            "energy_kwh_complete_day", "readings_observed",
            "readings_expected_in_span", "coverage_pct_in_span", "is_complete_day",
            F.when(F.col("is_complete_day"), F.col("day_peak.kwh"))
             .otherwise(F.lit(None).cast("double")).alias("peak_halfhour_kwh"),
            F.when(F.col("is_complete_day"),
                   F.floor(-F.col("day_peak.neg_slot") / F.lit(2)).cast("int"))
             .otherwise(F.lit(None).cast("int")).alias("peak_hour"),
            F.when(F.col("is_complete_day") & (F.col("energy_kwh_observed") > 0),
                   F.round(100.0 * F.col("evening_kwh") / F.col("energy_kwh_observed"), 4))
             .otherwise(F.lit(None).cast("double")).alias("evening_energy_share_pct"),
            F.when(F.col("is_complete_day") & (F.col("energy_kwh_observed") > 0),
                   F.round(100.0 * F.col("overnight_kwh") / F.col("energy_kwh_observed"), 4))
             .otherwise(F.lit(None).cast("double")).alias("overnight_energy_share_pct"),
        )
        .select(*DAILY_COLUMNS)
    )
    daily.repartition(4).write.mode("errorifexists").parquet(str(daily_output))
    home_hour.unpersist()

    # Check what was physically published, not just the pre-write plan.
    hp = spark.read.parquet(str(hour_output))
    dp = spark.read.parquet(str(daily_output))
    hs = hp.agg(F.count("*").alias("hours"),
                F.sum("readings_observed").alias("readings"),
                F.sum("energy_kwh_observed").alias("kwh")).first()
    ds = dp.agg(
        F.count("*").alias("days"),
        F.sum("readings_observed").alias("readings"),
        F.sum("readings_expected_in_span").alias("expected"),
        F.sum(F.when((F.col("readings_observed") < 0)
                     | (F.col("readings_observed") > F.col("readings_expected_in_span"))
                     | (F.col("readings_expected_in_span") > 48)
                     | (F.col("readings_expected_in_span") <= 0)
                     | (F.col("coverage_pct_in_span") > 100), 1).otherwise(0))
         .alias("invalid_days"),
        F.sum(F.when((F.col("readings_observed") == 0)
                     & (F.col("energy_kwh_observed").isNotNull()), 1).otherwise(0))
         .alias("zero_day_fake_energy"),
        F.sum(F.when((~F.col("is_complete_day"))
                     & F.col("energy_kwh_complete_day").isNotNull(), 1).otherwise(0))
         .alias("incomplete_day_fake_total"),
        F.sum("energy_kwh_observed").alias("kwh"),
    ).first()
    expected_readings = int(hh_stats["readings"])
    if (int(hs["readings"]) != expected_readings
            or int(ds["readings"]) != expected_readings
            or any(int(ds[k]) for k in ("invalid_days", "zero_day_fake_energy", "incomplete_day_fake_total"))):
        raise ValueError(f"Bucket {bucket:02d} readback reconciliation FAILED: {hs} vs {ds}")
    if abs(float(hs["kwh"]) - float(ds["kwh"])) > max(0.01, abs(float(hs["kwh"])) * 1e-9):
        raise ValueError(f"Bucket {bucket:02d}: hourly and daily kWh totals disagree")

    meta = {
        "version": VERSION, "bucket": bucket, "hourly_rows": int(hs["hours"]),
        "daily_rows": int(ds["days"]), "readings_observed": expected_readings,
        "readings_expected_in_span": int(ds["expected"]),
        "observed_kwh": float(ds["kwh"]),
        "hourly_columns": ["hour_key", "energy_kwh_observed", "energy_kwh_complete_homes",
                           "homes_reporting", "homes_complete", "readings_observed"],
        "daily_columns": DAILY_COLUMNS,
        "zero_reading_days_preserved": True,
    }
    atomic_json(BUCKET_MARKERS / f"bucket_{bucket:02d}.json", meta)
    if bucket_state(bucket) is None:
        raise AssertionError("Bucket checkpoint was not published")
    print(f"PASSED | Bucket {bucket:02d} | Readings: {expected_readings:,} | "
          f"Hour groups: {meta['hourly_rows']:,} | Home-days: {meta['daily_rows']:,}", flush=True)


def write_daily_final_stage():
    stage = WORK / "_publishing_eda_household_daily"
    if stage.exists() or FINAL_DAILY.exists():
        raise RuntimeError(f"Final daily output/stage already exists; inspect {stage} and {FINAL_DAILY}")
    stage.mkdir(parents=True)
    num_parts = 0
    # Copy independent already-validated Parquet part files. This avoids a
    # whole-dataset Spark shuffle, and NO 'bucket' column enters the schema.
    for b in range(BUCKETS):
        parts = sorted((bucket_dir(b) / "daily").glob("part-*.parquet"))
        if not parts:
            raise RuntimeError(f"Missing daily Parquet parts in bucket {b:02d}")
        for j, p in enumerate(parts):
            shutil.copy2(p, stage / f"part-b{b:02d}-{j:03d}.parquet")
            num_parts += 1
    (stage / "_SUCCESS").write_text("", encoding="utf-8")
    return stage, num_parts


def publish_final(spark, spans, total_span_expected, bucket_metas):
    print("\nAll 32 buckets validated. Publishing final two EDA datasets...", flush=True)
    if FINAL_HOURLY.exists() or FINAL_DAILY.exists() or GLOBAL_MARKER.exists():
        raise RuntimeError("Final EDA output/manifest already exists; inspect before rerun")
    global_hours = hour_expected_grid(spans)
    if sum(n for _, n in global_hours) != total_span_expected:
        raise AssertionError("Python hourly expected grid != household span total")
    expected_df = spark.createDataFrame(global_hours, ["hour_key", "readings_expected_in_span"])
    hour_paths = [str(bucket_dir(i) / "hourly") for i in range(BUCKETS)]
    hh = spark.read.parquet(*hour_paths)
    hourly_sum = (
        hh.groupBy("hour_key")
        .agg(
            F.sum("energy_kwh_observed").alias("energy_kwh_observed"),
            F.sum("energy_kwh_complete_homes").alias("energy_kwh_complete_homes"),
            F.sum("homes_reporting").cast("long").alias("homes_reporting"),
            F.sum("homes_complete").cast("long").alias("homes_complete"),
            F.sum("readings_observed").cast("long").alias("readings_observed"),
        )
    )
    final_hourly = (
        expected_df.join(hourly_sum, "hour_key", "left")
        .withColumn("readings_observed", F.coalesce("readings_observed", F.lit(0)).cast("long"))
        .withColumn("homes_reporting", F.coalesce("homes_reporting", F.lit(0)).cast("long"))
        .withColumn("homes_complete", F.coalesce("homes_complete", F.lit(0)).cast("long"))
        .withColumn("energy_kwh_complete_homes",
                    F.coalesce("energy_kwh_complete_homes", F.lit(0.0)).cast("double"))
        .withColumn("coverage_pct_in_span", F.when(
            F.col("readings_expected_in_span") > 0,
            F.round(100.0 * F.col("readings_observed") / F.col("readings_expected_in_span"), 4)
        ).otherwise(F.lit(None).cast("double")))
        .withColumn("avg_kwh_per_complete_home", F.when(
            F.col("homes_complete") > 0,
            F.col("energy_kwh_complete_homes") / F.col("homes_complete")
        ).otherwise(F.lit(None).cast("double")))
        .withColumn("hour_start", F.to_timestamp_ntz(
            F.concat(F.col("hour_key"), F.lit(":00:00"))
        ))
        .select(*HOURLY_COLUMNS)
    )
    stage_hour = WORK / "_publishing_eda_hourly_history"
    if stage_hour.exists():
        raise RuntimeError(f"Partial hourly staging folder exists: {stage_hour}")
    final_hourly.coalesce(2).write.mode("errorifexists").parquet(str(stage_hour))
    stage_daily, num_part_files = write_daily_final_stage()

    hp = spark.read.parquet(str(stage_hour))
    dp = spark.read.parquet(str(stage_daily))
    if hp.columns != HOURLY_COLUMNS or dp.columns != DAILY_COLUMNS:
        raise ValueError("Final schema column names/order differ from approved 9+13 columns")
    ha = hp.agg(
        F.count("*").alias("hours"),
        F.countDistinct("hour_start").alias("distinct_hours"),
        F.sum("readings_observed").alias("readings"),
        F.sum(F.when(F.col("readings_observed") > F.col("readings_expected_in_span"), 1)
              .otherwise(0)).alias("bad_coverage_hours"),
        F.sum(F.when(F.col("hour_start").isNull(), 1).otherwise(0)).alias("null_hours"),
        F.sum(F.when(F.col("readings_expected_in_span") > 0,
                     F.col("readings_expected_in_span")).otherwise(0)).alias("expected"),
        F.sum(F.when(F.year("hour_start") == 2013, F.col("readings_observed"))
              .otherwise(0)).alias("readings_2013"),
        F.sum("energy_kwh_observed").alias("kwh"),
    ).first()
    da = dp.agg(
        F.count("*").alias("days"),
        F.sum("readings_observed").alias("readings"),
        F.sum("readings_expected_in_span").alias("expected"),
        F.sum(F.when((~F.col("is_complete_day")) & F.col("energy_kwh_complete_day").isNotNull(), 1)
              .otherwise(0)).alias("bad_complete_day"),
        F.sum("energy_kwh_observed").alias("kwh"),
    ).first()
    expected_hours = len(global_hours)
    expected_by_checkpoints = sum(m["readings_observed"] for m in bucket_metas)
    expected_days = sum(m["daily_rows"] for m in bucket_metas)
    if (int(ha["hours"]) != expected_hours or int(ha["distinct_hours"]) != expected_hours
            or int(ha["readings"]) != EXPECTED_SILVER_READINGS
            or int(ha["readings_2013"]) != EXPECTED_2013_SILVER_READINGS
            or int(da["readings"]) != EXPECTED_SILVER_READINGS
            or int(da["days"]) != expected_days
            or int(ha["expected"]) != total_span_expected
            or int(da["expected"]) != total_span_expected
            or expected_by_checkpoints != EXPECTED_SILVER_READINGS
            or int(ha["bad_coverage_hours"]) != 0 or int(ha["null_hours"]) != 0
            or int(da["bad_complete_day"]) != 0):
        raise ValueError(f"Final EDA validation FAILED: hourly={ha}; daily={da}")
    if abs(float(ha["kwh"]) - float(da["kwh"])) > max(0.05, abs(float(ha["kwh"])) * 1e-9):
        raise ValueError("Final hourly/daily total kWh mismatch")
    # Check duplicates by natural key, and counts of individual households.
    distinct_day_keys = dp.select("household_id", "date").distinct().count()
    distinct_households = dp.select("household_id").distinct().count()
    if distinct_day_keys != int(da["days"]) or distinct_households != EXPECTED_SILVER_HOUSEHOLDS:
        raise ValueError("Duplicate home-date or household count mismatch in daily EDA output")

    # Publish only after verifying BOTH staged datasets.
    stage_hour.rename(FINAL_HOURLY)
    stage_daily.rename(FINAL_DAILY)
    result = {
        "version": VERSION,
        "source_silver_valid_rows": EXPECTED_SILVER_READINGS,
        "households": EXPECTED_SILVER_HOUSEHOLDS,
        "source_folders": EXPECTED_SOURCES,
        "merged_buckets": BUCKETS,
        "hourly_rows": int(ha["hours"]),
        "hourly_columns": HOURLY_COLUMNS,
        "daily_rows": int(da["days"]),
        "daily_columns": DAILY_COLUMNS,
        "observed_readings": int(ha["readings"]),
        "2013_readings_reconciled": int(ha["readings_2013"]),
        "expected_readings_observed_span": total_span_expected,
        "historical_coverage_pct_in_span": round(100 * EXPECTED_SILVER_READINGS / total_span_expected, 4),
        "daily_parquet_part_files": num_part_files,
        "coverage_assumption": "inclusive first-to-last valid reading per household; naive half-hour slots",
        "timestamp_assumption": "TIMESTAMP_NTZ; no UTC/BST localization",
        "no_imputation_or_outlier_removal": True,
        "missing_full_days_preserved": True,
        "household_day_zero_readings": "energy_kwh_observed=NULL; readings_observed=0",
        "zero_energy_complete_day_shares": "NULL because percentage of zero is undefined",
        "tariff_group_policy": "day's unique tariff, MIXED if multiple; on zero-observation days use household overall tariff (or MIXED/NULL)",
        "hourly_parquet": str(FINAL_HOURLY),
        "daily_parquet": str(FINAL_DAILY),
    }
    atomic_json(GLOBAL_MARKER, result)
    print("\nEDA FINAL DATASETS: PASSED!")
    print(f"Hourly history: {result['hourly_rows']:,} rows x {len(HOURLY_COLUMNS)} columns")
    print(f"Household daily: {result['daily_rows']:,} rows x {len(DAILY_COLUMNS)} columns")
    print(f"Unique households: {result['households']:,}")
    print(f"Observed readings reconciled: {result['observed_readings']:,}")
    print(f"Coverage in observed spans: {result['historical_coverage_pct_in_span']:.4f}%")
    print("Hourly Parquet:", FINAL_HOURLY)
    print("Daily Parquet:", FINAL_DAILY)
    print("Manifest:", GLOBAL_MARKER)


def main():
    source_buckets = get_source_paths()
    spans, expected_span_slots = parse_spans()
    print("=" * 68)
    print("STEP 26B - MERGE BUCKETS & BUILD TWO APPROVED EDA PARQUETS")
    print("=" * 68)
    print("Input: 168/168 Step 26A sources, 32 household hash buckets")
    print("Expected Silver readings:", f"{EXPECTED_SILVER_READINGS:,}")
    print("Distinct Silver households:", len(spans))
    print("Historical coverage: first-to-last observed span, no timezone conversion")
    print("Final schemas: 9 hourly columns + 13 household-daily columns")
    print("Raw/Bronze/Silver/2013 Gold: READ-ONLY")

    if GLOBAL_MARKER.exists() or FINAL_HOURLY.exists() or FINAL_DAILY.exists():
        if (GLOBAL_MARKER.is_file() and (FINAL_HOURLY / "_SUCCESS").is_file()
                and (FINAL_DAILY / "_SUCCESS").is_file()):
            meta = json.loads(GLOBAL_MARKER.read_text(encoding="utf-8"))
            if meta.get("version") == VERSION:
                print("FINAL EDA DATASETS already published. Nothing to rebuild.")
                print("Hourly Parquet:", FINAL_HOURLY)
                print("Daily Parquet:", FINAL_DAILY)
                return
        raise RuntimeError("Partial/foreign final EDA dataset found. Inspect before continuing")

    completed = [b for b in range(BUCKETS) if bucket_state(b) is not None]
    remaining = [b for b in range(BUCKETS) if b not in completed]
    selected = remaining if MAX_NEW_BUCKETS is None else remaining[:MAX_NEW_BUCKETS]
    print("Already completed buckets:", len(completed))
    print("Remaining buckets:", len(remaining))
    print("Selected now:", len(selected))
    spark = None
    try:
        if selected or not remaining:
            spark = make_spark()
            schema = T.StructType([
                T.StructField("household_id", T.StringType(), False),
                T.StructField("first_timestamp", T.StringType(), False),
                T.StructField("last_timestamp", T.StringType(), False),
                T.StructField("first_slot", T.IntegerType(), False),
                T.StructField("last_slot", T.IntegerType(), False),
            ])
            span_df = spark.createDataFrame(spans, schema).withColumn(
                "bucket", F.pmod(F.xxhash64("household_id"), F.lit(BUCKETS)).cast("int")
            )
            for b in selected:
                process_bucket(spark, b, source_buckets, span_df)

            done = [bucket_state(b) for b in range(BUCKETS)]
            print("\nEDA BUCKET MERGE PROGRESS")
            print(f"Completed: {sum(m is not None for m in done)}/{BUCKETS}")
            if all(m is not None for m in done):
                publish_final(spark, spans, expected_span_slots, done)
            else:
                print("Intermediate bucket merges saved. Set MAX_NEW_BUCKETS=None and rerun.")
                print("Bucket checkpoint folder:", BUCKET_MARKERS)
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
