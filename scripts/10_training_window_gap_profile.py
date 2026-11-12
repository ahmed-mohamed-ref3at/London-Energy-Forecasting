"""Step 21 — Train-only half-hour gap profiling (2013 Jan-Jun).

Uses Silver VALID only, reads one source at a time, and compresses each
household/day to a 48-bit mask. Across-source masks are OR-merged to handle
households whose data appear in multiple source CSV files. Then the script
counts actual missing half-hour slots, consecutive gap runs, and longest gaps.

The cohort must be chosen using training-period information ONLY. The full-
year 2013 report is exploratory and must not be used to select households
before evaluating on months within 2013.

The timeline is deliberately NAIVE (TIMESTAMP_NTZ); UTC/British-local time
semantics and DST need independent verification before Gold production.

First run MAX_NEW_FILES = 1, then set to None and rerun.

    python scripts/21_training_window_gap_profile.py
"""

import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE = Path(__file__).resolve().parents[1]
SILVER_ROOT = BASE / "data" / "silver" / "readings_by_file"
HOUSEHOLD_MANIFEST = BASE / "reports" / "household_coverage_summary.csv"
CHECKPOINT_DIR = BASE / "reports" / "training_2013H1_day_masks_by_file"
OUTPUT_CSV = BASE / "reports" / "training_2013H1_household_gaps.csv"
OUTPUT_JSON = BASE / "reports" / "training_2013H1_gap_overview.json"

# Run one source first. After it passes, change to None to finish all 168.
#MAX_NEW_FILES = 1
MAX_NEW_FILES = None

SOURCE_COUNT = 168
EXPECTED_SILVER_ROWS = 167_811_461
VERSION = "training_2013H1_daily_bitmask_v1"
START = datetime(2013, 1, 1)
END = datetime(2013, 7, 1)  # EXCLUSIVE; future validation is not used
HALF_HOUR = timedelta(minutes=30)
EXPECTED_DAYS = (END - START).days  # 181 days
EXPECTED_SLOTS = EXPECTED_DAYS * 48  # 8,688 half-hour readings
MAX_DAY_MASK = (1 << 48) - 1
assert EXPECTED_DAYS == 181 and EXPECTED_SLOTS == 8_688

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def atomic_json(path, payload):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def source_index(path):
    return int(path.name.rsplit("_", 1)[-1])


if not SILVER_ROOT.is_dir():
    raise FileNotFoundError(SILVER_ROOT)
if not HOUSEHOLD_MANIFEST.is_file():
    raise FileNotFoundError(HOUSEHOLD_MANIFEST)

sources = sorted(
    (p for p in SILVER_ROOT.iterdir() if p.is_dir()), key=source_index
)
if len(sources) != SOURCE_COUNT:
    raise ValueError(f"Expected 168 Silver sources, found {len(sources)}")

with HOUSEHOLD_MANIFEST.open("r", newline="", encoding="utf-8-sig") as handle:
    households = {}
    reader = csv.DictReader(handle)
    if "household_id" not in (reader.fieldnames or []):
        raise ValueError("Missing household_id in Step 17 manifest")
    for row in reader:
        hid = row["household_id"]
        if not hid or hid in households:
            raise ValueError(f"Invalid/duplicate household ID: {hid!r}")
        households[hid] = row
if len(households) != 5_561:
    raise ValueError(f"Expected 5,561 Silver households, found {len(households)}")

expected_by_source = {}
for source in sources:
    marker = source / "_silver_complete.json"
    success = source / "valid" / "_SUCCESS"
    if not marker.is_file() or not success.is_file():
        raise RuntimeError(f"Missing Silver completion marker: {source.name}")
    info = json.loads(marker.read_text(encoding="utf-8"))
    count = info.get("valid_rows")
    if (info.get("source_file") != source.name + ".csv"
            or type(count) is not int or count < 0):
        raise ValueError(f"Invalid Silver checkpoint: {source.name}")
    expected_by_source[source.name] = count

if sum(expected_by_source.values()) != EXPECTED_SILVER_ROWS:
    raise ValueError("Silver global row count differs from Step 16")


def checkpoint_path(source):
    return CHECKPOINT_DIR / (source.name + ".json")


def load_checkpoint(source):
    path = checkpoint_path(source)
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("version") != VERSION
            or report.get("source_file") != source.name + ".csv"
            or report.get("source_valid_rows") != expected_by_source[source.name]
            or report.get("window_start") != START.isoformat(sep=" ")
            or report.get("window_end_exclusive") != END.isoformat(sep=" ")
            or report.get("expected_slots") != EXPECTED_SLOTS):
        raise ValueError(f"Checkpoint settings mismatch: {path}")
    rows = report.get("daily_masks")
    if not isinstance(rows, list):
        raise ValueError(f"Missing daily_masks: {path}")
    total = 0
    seen = set()
    for item in rows:
        if not isinstance(item, list) or len(item) != 4:
            raise ValueError(f"Bad daily mask row: {path}")
        hid, day_text, mask, count = item
        if (hid not in households or type(day_text) is not str
                or type(mask) is not int or not 0 <= mask <= MAX_DAY_MASK
                or type(count) is not int or not 1 <= count <= 48
                or mask.bit_count() != count):
            raise ValueError(f"Invalid source day mask in {path}: {item}")
        day = date.fromisoformat(day_text)
        if not START.date() <= day < END.date():
            raise ValueError(f"Out of training window: {path}, {day}")
        key = (hid, day_text)
        if key in seen:
            raise ValueError(f"Repeated household/day in {path}: {key}")
        seen.add(key)
        total += count
    if (report.get("window_rows") != total
            or report.get("daily_groups") != len(rows)
            or total > expected_by_source[source.name]):
        raise ValueError(f"Per-source row reconciliation failed: {path}")
    return report


pending = [source for source in sources if load_checkpoint(source) is None]
selected = pending if MAX_NEW_FILES is None else pending[:MAX_NEW_FILES]

print("=" * 60)
print("LONDON SMART METER - TRAINING-ONLY TIME GAP PROFILE")
print("=" * 60)
print("Training window: [2013-01-01, 2013-07-01)")
print("Naive half-hour slots per household:", f"{EXPECTED_SLOTS:,}")
print("Total Silver sources:", len(sources))
print("Already completed:", len(sources) - len(pending))
print("Remaining:", len(pending))
print("Selected now:", len(selected))


def process(spark, source):
    print(f"\nProcessing: {source.name}", flush=True)
    df = (
        spark.read.parquet(str(source / "valid"))
        .select("household_id", "reading_timestamp")
        .filter(F.col("reading_timestamp") >=
                F.lit("2013-01-01 00:00:00").cast("timestamp_ntz"))
        .filter(F.col("reading_timestamp") <
                F.lit("2013-07-01 00:00:00").cast("timestamp_ntz"))
    )

    # Step 17 established zero off-grid readings for all Silver records.
    # Defensively reject clock times not aligned to :00/:30 before bitpacking.
    off_grid = df.filter(
        (~F.minute("reading_timestamp").isin(0, 30))
        | (F.second("reading_timestamp") != 0)
    ).limit(1).count()
    if off_grid:
        raise ValueError(f"Off-grid timestamp in {source.name}")

    indexed = (
        df.withColumn("reading_day", F.to_date("reading_timestamp"))
        .withColumn("slot_index", F.hour("reading_timestamp") * 2
                    + F.when(F.minute("reading_timestamp") == 30, 1)
                       .otherwise(0))
        # 2**47 fits in signed BIGINT; each power of 2 is exact as DOUBLE.
        .withColumn("slot_bit", F.pow(F.lit(2.0), F.col("slot_index")).cast("long"))
    )
    daily = (
        indexed.groupBy("household_id", "reading_day")
        .agg(F.bit_or("slot_bit").alias("day_mask"),
             F.count("*").alias("day_rows"))
    )

    # Only ~5,000-8,000 compact household/day masks per source reach Python.
    daily_masks = []
    total_rows = 0
    for row in daily.collect():
        hid, day, mask, n = (row["household_id"], row["reading_day"],
                              int(row["day_mask"]), int(row["day_rows"]))
        if hid not in households or not 1 <= n <= 48 or mask.bit_count() != n:
            raise ValueError(f"Bad/duplicate Silver slots: {hid}, {day}")
        daily_masks.append([hid, day.isoformat(), mask, n])
        total_rows += n

    daily_masks.sort(key=lambda r: (r[0], r[1]))
    report = {
        "version": VERSION,
        "source_file": source.name + ".csv",
        "source_valid_rows": expected_by_source[source.name],
        "window_start": START.isoformat(sep=" "),
        "window_end_exclusive": END.isoformat(sep=" "),
        "expected_slots": EXPECTED_SLOTS,
        "window_rows": total_rows,
        "daily_groups": len(daily_masks),
        "daily_masks": daily_masks,
    }
    atomic_json(checkpoint_path(source), report)
    # Verify the checkpoint before considering it complete.
    load_checkpoint(source)
    print(
        f"PASSED | Training rows: {total_rows:,} "
        f"| Household-day masks: {len(daily_masks):,}", flush=True
    )


if selected:
    spark = (
        SparkSession.builder.appName("LondonEnergyTrainingWindowGaps")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.timestampType", "TIMESTAMP_NTZ")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        for source in selected:
            process(spark, source)
    finally:
        spark.stop()

reports = [load_checkpoint(source) for source in sources]
finished = [report for report in reports if report is not None]
print("\nTRAINING GAP PROFILE PROGRESS")
print(f"Completed: {len(finished)}/{SOURCE_COUNT}")
if len(finished) != SOURCE_COUNT:
    print("Partial masks saved. Resume later.")
    sys.exit(0)

# Merge 48-bit day masks globally, so the same home's readings across
# multiple Silver sources are evaluated together without duplicate counting.
masks = defaultdict(int)
window_rows = 0
for report in finished:
    window_rows += report["window_rows"]
    for hid, day, mask, n in report["daily_masks"]:
        masks[(hid, day)] |= mask

observed_merged = sum(mask.bit_count() for mask in masks.values())
if observed_merged != window_rows:
    raise ValueError(
        "Cross-source duplicate/overlap after merging masks: "
        f"source rows={window_rows:,}, unique slots={observed_merged:,}"
    )

all_days = [START.date() + timedelta(days=i) for i in range(EXPECTED_DAYS)]
full_mask = (1 << EXPECTED_SLOTS) - 1
records = []
for hid in sorted(households):
    # Each day mask uses bit 0 = 00:00 and bit 47 = 23:30.
    # Append days in REVERSE chronological order. Consequently, bit 0 of
    # the final integer = first training slot and highest bit = last slot;
    # consecutive bits are consecutive half-hours even across midnight.
    observed_bits = 0
    for day in reversed(all_days):
        observed_bits = ((observed_bits << 48)
                         | masks.get((hid, day.isoformat()), 0))

    observed = observed_bits.bit_count()
    missing = EXPECTED_SLOTS - observed
    missing_bits = full_mask ^ observed_bits
    runs = [len(match.group()) for match in re.finditer(
        r"1+", format(missing_bits, f"0{EXPECTED_SLOTS}b")
    )]
    longest = max(runs, default=0)

    records.append({
        "household_id": hid,
        "observed_train_slots": observed,
        "expected_train_slots": EXPECTED_SLOTS,
        "missing_train_slots": missing,
        "train_coverage_pct": round(100 * observed / EXPECTED_SLOTS, 4),
        "gap_runs": len(runs),
        "longest_gap_slots": longest,
        "longest_gap_hours": longest / 2,
        "first_slot_observed": int(bool(observed_bits & 1)),
        "last_slot_observed": int(bool(observed_bits & (1 << (EXPECTED_SLOTS - 1)))),
    })

if sum(r["observed_train_slots"] for r in records) != window_rows:
    raise ValueError("Global household training window row reconciliation failed")

with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

thresholds = [90, 95, 99, 99.5, 99.9, 100]
counts = {
    str(threshold): sum(r["train_coverage_pct"] >= threshold for r in records)
    for threshold in thresholds
}
# Preliminary informational criteria, NOT an adopted or frozen cohort.
preliminary = sum(
    r["train_coverage_pct"] >= 99.5
    and r["longest_gap_hours"] <= 6
    and r["first_slot_observed"] == 1
    and r["last_slot_observed"] == 1
    for r in records
)
summary = {
    "version": VERSION,
    "time_basis": "naive half-hour TIMESTAMP_NTZ; UTC/BST not verified",
    "window_start_inclusive": START.isoformat(sep=" "),
    "window_end_exclusive": END.isoformat(sep=" "),
    "silver_valid_rows_in_training_window": window_rows,
    "households_in_silver": len(records),
    "expected_slots_per_household": EXPECTED_SLOTS,
    "median_training_coverage_pct": statistics.median(
        r["train_coverage_pct"] for r in records
    ),
    "coverage_threshold_counts_all_homes": counts,
    "preliminary_99_5pct_max_6h_both_endpoints_count": preliminary,
    "future_validation_included_in_selection": False,
    "note": "Exploratory criteria, NOT a frozen cohort. Revisit timezone/DST and splits.",
}
atomic_json(OUTPUT_JSON, summary)

print("\nFINAL TRAINING-ONLY TIME GAP RESULTS")
print("Training rows:", f"{window_rows:,}")
print("Households:", f"{len(records):,}")
print("Median training coverage:", f"{summary['median_training_coverage_pct']:.4f}%")
for k, value in counts.items():
    print(f"Coverage >= {k}%: {value:,}")
print("Preliminary >=99.5% and max gap <=6h and both endpoints:", f"{preliminary:,}")
print("TRAINING WINDOW ROW RECONCILIATION: PASSED!")
print("CSV:", OUTPUT_CSV)
print("Summary:", OUTPUT_JSON)
print("CAUTION: DST/timezone semantics still unverified.")
