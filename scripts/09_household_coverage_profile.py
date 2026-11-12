"""Step 17A: Build resumable household coverage profiles from Silver Valid Parquet.

Run from the project root:
    python scripts/17_household_coverage_profile.py

Start with MAX_NEW_FILES = 1, inspect the output, then set it to None.

This produces one small CSV + JSON checkpoint per source and, after all 168
sources have finished, a merged household coverage CSV and a JSON summary.
It does not modify any Bronze or Silver files.

IMPORTANT: Expected half-hour slots are *naive wall-clock estimates*.
Do not treat the estimated gaps as confirmed physical gaps until timezone,
DST, off-grid timestamps, and consecutive-timestamp gaps have been audited.
"""

import csv
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE = Path(__file__).resolve().parents[1]
SILVER_ROOT = BASE / "data" / "silver" / "readings_by_file"
PARTIAL_DIR = BASE / "reports" / "household_coverage_by_file"
HOUSEHOLD_REPORT = BASE / "reports" / "household_coverage_summary.csv"
SUMMARY_JSON = BASE / "reports" / "household_coverage_overview.json"

# PILOT: process only the first source. After it passes, change to None.
#MAX_NEW_FILES = 1
MAX_NEW_FILES = None
EXPECTED_SOURCE_COUNT = 168
EXPECTED_SILVER_ROWS = 167_811_461
VERSION = "coverage_v1"

PROFILE_COLUMNS = [
    "household_id",
    "observed_readings",
    "first_timestamp",
    "last_timestamp",
    "off_grid_readings",
]

PARTIAL_DIR.mkdir(parents=True, exist_ok=True)


def source_number(folder):
    return int(folder.name.rsplit("_", 1)[-1])


sources = sorted(
    [p for p in SILVER_ROOT.iterdir() if p.is_dir()],
    key=source_number,
)
if len(sources) != EXPECTED_SOURCE_COUNT:
    raise ValueError(f"Expected 168 Silver source folders, got {len(sources)}")


def silver_metadata(source):
    path = source / "_silver_complete.json"
    if not path.is_file():
        raise ValueError(f"Missing Silver checkpoint: {path}")
    if not (source / "valid" / "_SUCCESS").is_file():
        raise ValueError(f"Missing Silver Valid _SUCCESS: {source.name}")
    info = json.loads(path.read_text(encoding="utf-8"))
    if info.get("source_file") != source.name + ".csv":
        raise ValueError(f"Silver checkpoint source mismatch: {source.name}")
    if not isinstance(info.get("valid_rows"), int):
        raise ValueError(f"Missing valid_rows in {path}")
    return info


expected_valid_rows = {source.name: silver_metadata(source)["valid_rows"] for source in sources}
if sum(expected_valid_rows.values()) != EXPECTED_SILVER_ROWS:
    raise ValueError("Silver checkpoints do not total 167,811,461 rows")


def csv_path(source):
    return PARTIAL_DIR / f"{source.name}.csv"


def checkpoint_path(source):
    return PARTIAL_DIR / f"{source.name}.json"


def completed(source):
    csv_file = csv_path(source)
    marker = checkpoint_path(source)
    if not csv_file.exists() and not marker.exists():
        return None
    if not csv_file.is_file() or not marker.is_file():
        raise RuntimeError(f"Incomplete coverage output: {source.name}")
    result = json.loads(marker.read_text(encoding="utf-8"))
    if (
        result.get("version") != VERSION
        or result.get("source_file") != source.name + ".csv"
        or result.get("valid_rows") != expected_valid_rows[source.name]
    ):
        raise ValueError(f"Invalid coverage checkpoint: {source.name}")
    return result


pending = [source for source in sources if completed(source) is None]
selected = pending if MAX_NEW_FILES is None else pending[:MAX_NEW_FILES]

print("=" * 55)
print("LONDON SMART METER - HOUSEHOLD COVERAGE PROFILE")
print("=" * 55)
print("Total files:", len(sources))
print("Already completed:", len(sources) - len(pending))
print("Remaining:", len(pending))
print("Selected now:", len(selected))


def save_csv_atomic(target, records):
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=PROFILE_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(target)


def save_json_atomic(target, payload):
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(target)


def process_one(spark, source):
    print(f"\nProcessing: {source.name}", flush=True)
    df = spark.read.parquet(str(source / "valid")).select(
        "household_id", "reading_timestamp"
    )

    # A normal reading should sit at :00 or :30 with zero seconds/microseconds.
    # This is a format/grid check, NOT a DST or timezone decision.
    on_grid = (
        F.minute("reading_timestamp").isin(0, 30)
        & (F.second("reading_timestamp") == 0)
        & (F.date_format("reading_timestamp", "SSSSSS") == "000000")
    )

    grouped = (
        df.groupBy("household_id")
        .agg(
            F.count("*").alias("observed_readings"),
            F.min("reading_timestamp").alias("first_timestamp"),
            F.max("reading_timestamp").alias("last_timestamp"),
            F.sum(F.when(~on_grid, 1).otherwise(0)).alias("off_grid_readings"),
        )
        .orderBy("household_id")
    )

    # At most one record per household (5,566 global), NOT millions of rows.
    group_rows = grouped.collect()
    records = []
    for item in group_rows:
        if item["household_id"] is None or item["first_timestamp"] is None:
            raise ValueError(f"Unexpected null valid key: {source.name}")
        records.append(
            {
                "household_id": item["household_id"],
                "observed_readings": int(item["observed_readings"]),
                "first_timestamp": item["first_timestamp"].isoformat(sep=" "),
                "last_timestamp": item["last_timestamp"].isoformat(sep=" "),
                "off_grid_readings": int(item["off_grid_readings"]),
            }
        )

    source_total = sum(row["observed_readings"] for row in records)
    expected = expected_valid_rows[source.name]
    if source_total != expected:
        raise ValueError(
            f"Readings mismatch {source.name}: expected={expected}, got={source_total}"
        )

    # Marker is published only after the CSV has been written.
    save_csv_atomic(csv_path(source), records)
    result = {
        "version": VERSION,
        "source_file": source.name + ".csv",
        "valid_rows": source_total,
        "household_groups": len(records),
        "off_grid_readings": sum(r["off_grid_readings"] for r in records),
    }
    save_json_atomic(checkpoint_path(source), result)
    print(
        f"PASSED | Valid: {source_total:,} | Households: {len(records):,} "
        f"| Off-grid: {result['off_grid_readings']:,}",
        flush=True,
    )


def global_summary():
    # Each per-file CSV is compact; aggregate with standard Python (no large Spark shuffle).
    households = {}
    totals_by_files = 0
    for source in sources:
        checkpoint = completed(source)
        if checkpoint is None:
            raise RuntimeError("Global summary requested before all files completed")

        local_count = 0
        local_groups = 0
        local_off_grid = 0
        with csv_path(source).open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != PROFILE_COLUMNS:
                raise ValueError(f"Coverage CSV schema mismatch: {source.name}")
            for row in reader:
                household_id = row["household_id"]
                observed = int(row["observed_readings"])
                off_grid = int(row["off_grid_readings"])
                first = datetime.fromisoformat(row["first_timestamp"])
                last = datetime.fromisoformat(row["last_timestamp"])
                if observed < 1 or off_grid > observed or first > last:
                    raise ValueError(f"Invalid coverage metrics: {source.name}")
                local_count += observed
                local_groups += 1
                local_off_grid += off_grid
                if household_id not in households:
                    households[household_id] = {
                        "household_id": household_id,
                        "observed_readings": 0,
                        "first_timestamp": first,
                        "last_timestamp": last,
                        "off_grid_readings": 0,
                        "source_files_present": 0,
                    }
                current = households[household_id]
                current["observed_readings"] += observed
                current["first_timestamp"] = min(current["first_timestamp"], first)
                current["last_timestamp"] = max(current["last_timestamp"], last)
                current["off_grid_readings"] += off_grid
                current["source_files_present"] += 1
        if local_count != checkpoint["valid_rows"]:
            raise ValueError(f"Coverage CSV row sum mismatch: {source.name}")
        if local_groups != checkpoint["household_groups"]:
            raise ValueError(f"Coverage CSV groups mismatch: {source.name}")
        if local_off_grid != checkpoint["off_grid_readings"]:
            raise ValueError(f"Coverage CSV off-grid mismatch: {source.name}")
        totals_by_files += local_count

    if totals_by_files != EXPECTED_SILVER_ROWS:
        raise ValueError("Global Silver/coverage reconciliation FAILED")

    output_columns = [
        "household_id", "observed_readings", "first_timestamp",
        "last_timestamp", "source_files_present", "off_grid_readings",
        "expected_half_hours_wall_clock", "estimated_missing_slots_wall_clock",
        "coverage_pct_wall_clock",
    ]
    output_rows = []
    for household_id in sorted(households):
        x = households[household_id]
        duration_seconds = (x["last_timestamp"] - x["first_timestamp"]).total_seconds()
        aligned_span = (duration_seconds % 1800 == 0)
        valid_grid = x["off_grid_readings"] == 0 and aligned_span
        expected = int(duration_seconds / 1800) + 1 if valid_grid else None
        # This is only a *nominal wall-clock* estimate, not a DST-aware gap count.
        missing = max(expected - x["observed_readings"], 0) if expected is not None else None
        coverage = (
            round(100 * x["observed_readings"] / expected, 4)
            if expected is not None and expected > 0 else None
        )
        output_rows.append({
            "household_id": household_id,
            "observed_readings": x["observed_readings"],
            "first_timestamp": x["first_timestamp"].isoformat(sep=" "),
            "last_timestamp": x["last_timestamp"].isoformat(sep=" "),
            "source_files_present": x["source_files_present"],
            "off_grid_readings": x["off_grid_readings"],
            "expected_half_hours_wall_clock": expected if expected is not None else "",
            "estimated_missing_slots_wall_clock": missing if missing is not None else "",
            "coverage_pct_wall_clock": coverage if coverage is not None else "",
        })

    temp = HOUSEHOLD_REPORT.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(output_rows)
    temp.replace(HOUSEHOLD_REPORT)

    coverage_values = [
        float(row["coverage_pct_wall_clock"])
        for row in output_rows if row["coverage_pct_wall_clock"] != ""
    ]
    summary = {
        "source_files": len(sources),
        "silver_valid_rows": totals_by_files,
        "distinct_households": len(households),
        "off_grid_readings": sum(x["off_grid_readings"] for x in households.values()),
        "households_with_nominal_coverage_estimate": len(coverage_values),
        "median_coverage_pct_wall_clock": round(statistics.median(coverage_values), 4)
        if coverage_values else None,
        "interpretation_warning": (
            "Wall-clock estimates are preliminary, not exact gap counts. "
            "Confirm timestamp timezone/DST and consecutive-timestamp gaps before cohort selection."
        ),
    }
    save_json_atomic(SUMMARY_JSON, summary)

    print("\nFINAL GLOBAL HOUSEHOLD COVERAGE PROFILE")
    print("Files:", summary["source_files"])
    print("Silver valid rows:", f"{summary['silver_valid_rows']:,}")
    print("Distinct households:", f"{summary['distinct_households']:,}")
    print("Off-grid readings:", f"{summary['off_grid_readings']:,}")
    print("Median nominal coverage (%):", summary["median_coverage_pct_wall_clock"])
    print("GLOBAL COVERAGE ROW RECONCILIATION: PASSED!")
    print("Household report:", HOUSEHOLD_REPORT)
    print("Summary report:", SUMMARY_JSON)


if selected:
    spark = (
        SparkSession.builder
        .appName("LondonEnergyHouseholdCoverageProfile")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.timestampType", "TIMESTAMP_NTZ")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        for source in selected:
            process_one(spark, source)
    finally:
        spark.stop()

finished = sum(completed(source) is not None for source in sources)
print("\nCOVERAGE PROFILE PROGRESS")
print(f"Completed: {finished}/{EXPECTED_SOURCE_COUNT}")
if finished == EXPECTED_SOURCE_COUNT:
    global_summary()
else:
    print("Partial per-file profiles saved. Resume later.")
