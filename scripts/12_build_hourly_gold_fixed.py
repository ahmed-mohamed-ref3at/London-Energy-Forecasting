"""Step 24 - Build resumable, missingness-aware 2013 hourly Gold.

Input:
  data/silver/readings_by_file/<source>/valid/ (168 Parquet datasets)
  reports/cohorts/2013H1_candidate_v1/household_ids.txt
  reports/cohorts/2013H1_candidate_v1/manifest.json

Output after ALL sources have passed:
  data/gold/hourly_2013_cohort_v1/ (Parquet, exactly 8,760 hourly rows)
  reports/gold_hourly_2013_profile.csv
  reports/gold_hourly_2013_summary.json

The 5,104-home cohort was chosen with 2013 Jan-Jun readings only.
Jul-Dec data are aggregated for subsequent evaluation, not cohort selection.

Only the original naive timestamp grid is used: TIMESTAMP_NTZ, NO timezone
conversion. Hourly totals are sums of the *observed* half-hour kWh values.
Incomplete hours are flagged; complete_total_kwh is NULL unless the hour
contains both half-hour readings for ALL 5,104 cohort homes. Do not mistake
observed_total_kwh on an incomplete hour for complete household consumption.

This fixed version avoids Spark date_trunc on a TIMESTAMP_NTZ column by
grouping on a wall-clock hour string extracted from the original timestamp.
Start with MAX_NEW_FILES = 1, then set it to None and rerun.
    python scripts/24_build_hourly_gold_fixed.py
"""

import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType, StructField, TimestampNTZType, DoubleType,
    LongType, BooleanType,
)

BASE = Path(__file__).resolve().parents[1]
SILVER = BASE / "data" / "silver" / "readings_by_file"
COHORT_DIR = BASE / "reports" / "cohorts" / "2013H1_candidate_v1"
IDS_PATH = COHORT_DIR / "household_ids.txt"
MANIFEST_PATH = COHORT_DIR / "manifest.json"
PARTIAL = BASE / "reports" / "gold_hourly_2013_by_file"
GOLD_ROOT = BASE / "data" / "gold"
GOLD_FINAL = GOLD_ROOT / "hourly_2013_cohort_v1"
GOLD_STAGE = GOLD_ROOT / "_hourly_2013_cohort_v1_staging"
PROFILE_CSV = BASE / "reports" / "gold_hourly_2013_profile.csv"
SUMMARY_JSON = BASE / "reports" / "gold_hourly_2013_summary.json"

# Pilot one source file. After success: MAX_NEW_FILES = None.
#MAX_NEW_FILES = 1
MAX_NEW_FILES = None
VERSION = "hourly_2013_fixed_cohort_v2_wallclock_hour_key"
START = datetime(2013, 1, 1)
END = datetime(2014, 1, 1)  # exclusive
HOURS = int((END - START).total_seconds() / 3600)  # 8,760
EXPECTED_SOURCES = 168
EXPECTED_COHORT_HOMES = 5_104
EXPECTED_READINGS_PER_HOUR = EXPECTED_COHORT_HOMES * 2  # 10,208
EXPECTED_SILVER_ALL = 167_811_461
EXPECTED_SILVER_2013_ALL = 93_024_217
PARTIAL_FIELDS = ["hour_start", "observed_readings", "observed_kwh"]
PROFILE_FIELDS = [
    "hour_start", "observed_total_kwh", "complete_total_kwh",
    "observed_readings", "expected_readings", "missing_readings",
    "coverage_pct", "is_complete",
]


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def sha_file(path):
    return digest(path.read_bytes())


def safe_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def source_number(path):
    return int(path.name.rsplit("_", 1)[-1])


def make_spark():
    s = (
        SparkSession.builder.appName("LondonEnergyHourlyGold2013")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.timestampType", "TIMESTAMP_NTZ")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    s.sparkContext.setLogLevel("ERROR")
    return s


def hour_index(dt):
    if dt.tzinfo is not None or dt.minute or dt.second or dt.microsecond:
        raise ValueError(f"Invalid hourly timestamp: {dt}")
    seconds = (dt - START).total_seconds()
    if not (0 <= seconds < HOURS * 3600 and seconds % 3600 == 0):
        raise ValueError(f"Out-of-window hourly timestamp: {dt}")
    return int(seconds // 3600)


if not IDS_PATH.is_file() or not MANIFEST_PATH.is_file():
    raise FileNotFoundError("Step 22 cohort is missing")
ids_raw = IDS_PATH.read_bytes()
ids = [v.strip() for v in ids_raw.decode("utf-8-sig").splitlines() if v.strip()]
cohort_hash = digest(ids_raw)
cohort_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
if (len(ids) != EXPECTED_COHORT_HOMES or len(set(ids)) != len(ids)
        or cohort_manifest.get("households_eligible") != EXPECTED_COHORT_HOMES
        or cohort_manifest.get("household_ids_sha256") != cohort_hash
        or cohort_manifest.get("selection_uses_future_validation_or_test_data") is not False):
    raise ValueError("Candidate cohort count/checksum/train-only rule mismatch")

sources = sorted((p for p in SILVER.iterdir() if p.is_dir()), key=source_number)
if len(sources) != EXPECTED_SOURCES:
    raise ValueError(f"Expected {EXPECTED_SOURCES} Silver source folders, got {len(sources)}")

valid_by_source = {}
for source in sources:
    info_path = source / "_silver_complete.json"
    if not info_path.is_file() or not (source / "valid" / "_SUCCESS").is_file():
        raise ValueError(f"Incomplete Silver source: {source.name}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    count = info.get("valid_rows")
    if (info.get("source_file") != source.name + ".csv"
            or type(count) is not int or count < 0):
        raise ValueError(f"Invalid Silver checkpoint: {source.name}")
    valid_by_source[source.name] = count
if sum(valid_by_source.values()) != EXPECTED_SILVER_ALL:
    raise ValueError("Silver global source counts differ from Step 16")

PARTIAL.mkdir(parents=True, exist_ok=True)


def parts(source):
    return PARTIAL / (source.name + ".csv"), PARTIAL / (source.name + ".json")


def checked_checkpoint(source):
    csv_path, meta_path = parts(source)
    if not csv_path.exists() and not meta_path.exists():
        return None
    if not csv_path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"Incomplete partial output; inspect files for {source.name}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta.get("version") != VERSION
            or meta.get("source") != source.name
            or meta.get("silver_valid_rows") != valid_by_source[source.name]
            or meta.get("cohort_sha256") != cohort_hash
            or meta.get("window") != "2013-01-01/2014-01-01"
            or meta.get("csv_sha256") != sha_file(csv_path)):
        raise ValueError(f"Partial output checkpoint mismatch: {source.name}")
    return meta


pending = [p for p in sources if checked_checkpoint(p) is None]
selected = pending if MAX_NEW_FILES is None else pending[:MAX_NEW_FILES]

print("=" * 60)
print("STEP 24 - HOURLY GOLD, FROZEN COHORT (2013)")
print("=" * 60)
print("Window: [2013-01-01, 2014-01-01)")
print("Fixed candidate cohort:", len(ids))
print("Expected naive half-hour readings per hour:", EXPECTED_READINGS_PER_HOUR)
print("Total sources:", len(sources))
print("Already completed:", len(sources) - len(pending))
print("Remaining:", len(pending))
print("Selected now:", len(selected))
print("Timezone conversion: NONE (TIMESTAMP_NTZ)")


def process_one(spark, source, cohort_df):
    print(f"\nProcessing: {source.name}", flush=True)
    src = (
        spark.read.parquet(str(source / "valid"))
        .select("household_id", "reading_timestamp", "kwh")
        .filter(F.col("reading_timestamp") >=
                F.lit("2013-01-01 00:00:00").cast("timestamp_ntz"))
        .filter(F.col("reading_timestamp") <
                F.lit("2014-01-01 00:00:00").cast("timestamp_ntz"))
    )
    # Important: reading_timestamp is TIMESTAMP_NTZ (naive wall-clock time).
    # Do not pass it through date_trunc("hour", ...): Spark can change temporal
    # types during this expression. On this Windows installation the original
    # date_trunc grouping returned two rows for the SAME collected hour.
    # Casting NTZ to a string keeps the original calendar hour unconverted.
    # YYYY-MM-DD HH sorts chronologically without any timezone interpretation.
    hourly = (
        src.join(F.broadcast(cohort_df), "household_id", "inner")
        .withColumn("hour_key", F.substring(F.col("reading_timestamp").cast("string"), 1, 13))
        .groupBy("hour_key")
        .agg(
            F.count(F.lit(1)).alias("observed_readings"),
            F.sum("kwh").alias("observed_kwh"),
        )
        .orderBy("hour_key")
    )
    rows = hourly.collect()  # <= 8,760 tiny hourly groups, not raw rows
    records = []
    summed_rows = 0
    last_idx = -1
    for row in rows:
        hour_key = row["hour_key"]
        if not isinstance(hour_key, str) or len(hour_key) != 13:
            raise ValueError(f"Unexpected timestamp hour key: {hour_key!r}")
        try:
            ts = datetime.strptime(hour_key, "%Y-%m-%d %H")
        except ValueError as exc:
            raise ValueError(f"Invalid wall-clock hour key: {hour_key!r}") from exc
        n = int(row["observed_readings"])
        val = float(row["observed_kwh"])
        idx = hour_index(ts)
        if (idx <= last_idx or n <= 0 or n > EXPECTED_READINGS_PER_HOUR
                or not math.isfinite(val) or val < 0):
            raise ValueError(f"Invalid hourly partial: {source.name} / {ts}")
        records.append({
            "hour_start": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "observed_readings": str(n),
            "observed_kwh": repr(val),
        })
        summed_rows += n
        last_idx = idx
    if summed_rows > valid_by_source[source.name]:
        raise ValueError(f"Cohort rows exceed all Silver rows: {source.name}")

    csv_path, meta_path = parts(source)
    if csv_path.exists() or meta_path.exists():
        raise RuntimeError(f"Existing partial output needs inspection: {source.name}")
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    if tmp.exists():
        raise RuntimeError(f"Leftover temporary file (inspect it first): {tmp}")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARTIAL_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    tmp.replace(csv_path)
    meta = {
        "version": VERSION,
        "source": source.name,
        "silver_valid_rows": valid_by_source[source.name],
        "cohort_sha256": cohort_hash,
        "window": "2013-01-01/2014-01-01",
        "hour_groups": len(records),
        "cohort_readings": summed_rows,
        "csv_sha256": sha_file(csv_path),
    }
    safe_json(meta_path, meta)
    checked_checkpoint(source)
    print(f"PASSED | Cohort readings: {summed_rows:,} | Hour groups: {len(records):,}", flush=True)


spark = None
try:
    if selected:
        spark = make_spark()
        cohort_df = spark.createDataFrame([(hid,) for hid in ids], ["household_id"])
        for source in selected:
            process_one(spark, source, cohort_df)

    finished = [checked_checkpoint(s) for s in sources]
    done = [v for v in finished if v is not None]
    print("\nHOURLY GOLD PROGRESS")
    print(f"Completed: {len(done)}/{EXPECTED_SOURCES}")
    if len(done) < EXPECTED_SOURCES:
        print("Partial hourly data saved. Resume later.")
    else:
        print("\nCombining source-level hourly aggregates (tiny files, not raw readings)...")
        total_from_checkpoints = sum(x["cohort_readings"] for x in done)
        if not 0 < total_from_checkpoints <= EXPECTED_SILVER_2013_ALL:
            raise ValueError("Unexpected cohort row count relative to 2013 Silver")
        combined = [[0, 0.0] for _ in range(HOURS)]
        for source, checkpoint in zip(sources, finished):
            partial_file, _ = parts(source)
            seen = set()
            part_rows = 0
            part_groups = 0
            with partial_file.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != PARTIAL_FIELDS:
                    raise ValueError(f"Partial CSV columns mismatch: {source.name}")
                for entry in reader:
                    dt = datetime.strptime(entry["hour_start"], "%Y-%m-%d %H:%M:%S")
                    idx = hour_index(dt)
                    if idx in seen:
                        raise ValueError(f"Repeated hour in partial source: {source.name} / {dt}")
                    seen.add(idx)
                    n = int(entry["observed_readings"])
                    kwh = float(entry["observed_kwh"])
                    if (not 0 < n <= EXPECTED_READINGS_PER_HOUR
                            or not math.isfinite(kwh) or kwh < 0):
                        raise ValueError(f"Invalid partial values in {source.name}")
                    combined[idx][0] += n
                    combined[idx][1] = math.fsum((combined[idx][1], kwh))
                    part_rows += n
                    part_groups += 1
            if (part_rows != checkpoint["cohort_readings"]
                    or part_groups != checkpoint["hour_groups"]):
                raise ValueError(f"Partial CSV does not reconcile: {source.name}")

        output_rows = []
        profile_rows = []
        complete = 0
        with_any = 0
        missing_all = 0
        total_observed = 0
        for i, (n, summed_kwh) in enumerate(combined):
            if not 0 <= n <= EXPECTED_READINGS_PER_HOUR:
                raise ValueError(f"A combined hour exceeded the expected reading count: {i}")
            hour = START + timedelta(hours=i)
            is_complete = n == EXPECTED_READINGS_PER_HOUR
            covered = round(100.0 * n / EXPECTED_READINGS_PER_HOUR, 6)
            observed_kwh = summed_kwh if n else None
            complete_kwh = observed_kwh if is_complete else None
            missing = EXPECTED_READINGS_PER_HOUR - n
            output_rows.append((
                hour, observed_kwh, complete_kwh, n,
                EXPECTED_READINGS_PER_HOUR, missing, covered, is_complete,
            ))
            profile_rows.append({
                "hour_start": hour.strftime("%Y-%m-%d %H:%M:%S"),
                "observed_total_kwh": observed_kwh if n else "",
                "complete_total_kwh": complete_kwh if is_complete else "",
                "observed_readings": n,
                "expected_readings": EXPECTED_READINGS_PER_HOUR,
                "missing_readings": missing,
                "coverage_pct": covered,
                "is_complete": is_complete,
            })
            complete += int(is_complete)
            with_any += int(n > 0)
            missing_all += missing
            total_observed += n
        if total_observed != total_from_checkpoints:
            raise ValueError("Global hourly aggregation count reconciliation failed")

        GOLD_ROOT.mkdir(parents=True, exist_ok=True)
        if GOLD_STAGE.exists():
            raise RuntimeError(f"Leftover Gold staging path; inspect it: {GOLD_STAGE}")
        if GOLD_FINAL.exists():
            raise RuntimeError(f"Gold output exists; do not silently overwrite: {GOLD_FINAL}")
        if spark is None:
            spark = make_spark()
        schema = StructType([
            StructField("hour_start", TimestampNTZType(), False),
            StructField("observed_total_kwh", DoubleType(), True),
            StructField("complete_total_kwh", DoubleType(), True),
            StructField("observed_readings", LongType(), False),
            StructField("expected_readings", LongType(), False),
            StructField("missing_readings", LongType(), False),
            StructField("coverage_pct", DoubleType(), False),
            StructField("is_complete", BooleanType(), False),
        ])
        gold_df = spark.createDataFrame(output_rows, schema=schema)
        gold_df.coalesce(1).write.mode("errorifexists").parquet(str(GOLD_STAGE))
        check = spark.read.parquet(str(GOLD_STAGE)).agg(
            F.count("*").alias("hours"),
            F.sum("observed_readings").alias("rows"),
            F.sum("missing_readings").alias("missing"),
            F.sum(F.col("is_complete").cast("long")).alias("complete"),
        ).first()
        if (check["hours"] != HOURS or check["rows"] != total_observed
                or check["missing"] != missing_all or check["complete"] != complete):
            raise ValueError("Staged Gold Parquet validation failed")
        GOLD_STAGE.rename(GOLD_FINAL)

        with PROFILE_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDS)
            writer.writeheader()
            writer.writerows(profile_rows)

        summary = {
            "version": VERSION,
            "status": "PROVISIONAL - source timestamp timezone is unverified",
            "cohort_id": cohort_manifest.get("cohort_id"),
            "cohort_size": len(ids),
            "cohort_sha256": cohort_hash,
            "cohort_selected_using_only_2013H1": True,
            "window_start": "2013-01-01 00:00:00",
            "window_end_exclusive": "2014-01-01 00:00:00",
            "naive_timestamp_no_timezone_conversion": True,
            "source_count": len(sources),
            "hours": HOURS,
            "expected_readings_per_hour": EXPECTED_READINGS_PER_HOUR,
            "observed_cohort_readings": total_observed,
            "missing_cohort_half_hour_readings": missing_all,
            "hours_with_any_readings": with_any,
            "completely_observed_hours": complete,
            "incomplete_hours": HOURS - complete,
            "parquet": str(GOLD_FINAL.relative_to(BASE)).replace("\\", "/"),
            "observed_total_kwh_semantics": "sum of available household half-hour kWh; incomplete hours are lower bounds",
            "complete_total_kwh_semantics": "NULL unless all 10,208 readings are available in the hour",
            "no_missing_value_imputation": True,
        }
        safe_json(SUMMARY_JSON, summary)
        print("\nFINAL HOURLY GOLD RESULTS")
        print("Cohort households:", f"{len(ids):,}")
        print("Hours:", f"{HOURS:,}")
        print("Cohort half-hour readings in 2013:", f"{total_observed:,}")
        print("Complete hours:", f"{complete:,}")
        print("Incomplete hours:", f"{HOURS - complete:,}")
        print("Hour count/row count reconciliation: PASSED")
        print("Gold Parquet:", GOLD_FINAL)
        print("Hourly profile:", PROFILE_CSV)
        print("Summary:", SUMMARY_JSON)
        print("IMPORTANT: Incomplete hours have NULL complete_total_kwh.")
        print("IMPORTANT: Source timezone semantics remain unresolved.")
finally:
    if spark is not None:
        spark.stop()
