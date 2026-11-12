"""Step 25 — independent validation and coverage diagnostics for the 2013 Gold hourly layer.

Run from the project root:
    python scripts/25_gold_hourly_validation.py

Reads only the small final hourly Parquet and the source-hour checkpoint CSVs,
NOT the ~168 million Silver/Raw readings. Does not modify Gold.

Prerequisites: pandas, numpy, pyarrow (installed in the london-energy environment).
The 2013 cohort is PROVISIONAL until source timezone semantics are established.
"""

import csv
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
GOLD = BASE / "data" / "gold" / "hourly_2013_cohort_v1"
PROFILE = BASE / "reports" / "gold_hourly_2013_profile.csv"
SUMMARY = BASE / "reports" / "gold_hourly_2013_summary.json"
PARTIAL = BASE / "reports" / "gold_hourly_2013_by_file"
COHORT = BASE / "reports" / "cohorts" / "2013H1_candidate_v1"
MANIFEST = COHORT / "manifest.json"
IDS = COHORT / "household_ids.txt"
OUT_JSON = BASE / "reports" / "gold_hourly_2013_validation.json"
OUT_MONTHLY = BASE / "reports" / "gold_hourly_2013_validation_monthly.csv"

START = datetime(2013, 1, 1)
END = datetime(2014, 1, 1)
HOURS = 8760
HOMES = 5104
EXPECTED_PER_HOUR = 2 * HOMES
SOURCE_COUNT = 168
GOLD_FIELDS = [
    "hour_start", "observed_total_kwh", "complete_total_kwh",
    "observed_readings", "expected_readings", "missing_readings",
    "coverage_pct", "is_complete",
]
PARTIAL_FIELDS = ["hour_start", "observed_readings", "observed_kwh"]
VERSION = "hourly_2013_fixed_cohort_v2_wallclock_hour_key"


def require(ok, message):
    if not bool(ok):
        raise ValueError("VALIDATION FAILED: " + message)


def read_json(path):
    require(path.is_file(), f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def close_values(a, b, *, atol=1e-6, rtol=1e-10):
    return np.allclose(
        np.asarray(a, dtype="float64"),
        np.asarray(b, dtype="float64"),
        atol=atol, rtol=rtol, equal_nan=True,
    )


def main():
    print("=" * 65)
    print("STEP 25 - INDEPENDENT 2013 HOURLY GOLD VALIDATION")
    print("=" * 65)
    print("Gold:", GOLD)
    print("Cohort:", COHORT)

    require(GOLD.is_dir(), "Step 24 Gold Parquet directory is missing")
    require((GOLD / "_SUCCESS").is_file(), "Gold Parquet _SUCCESS is missing")
    require(PROFILE.is_file(), "Step 24 hourly CSV profile is missing")
    require(IDS.is_file(), "Step 22 household IDs file is missing")
    cohort_manifest = read_json(MANIFEST)
    summary = read_json(SUMMARY)

    # Ensure exactly the original 5,104 train-selected household IDs were used.
    cohort_raw = IDS.read_bytes()
    cohort_sha = hashlib.sha256(cohort_raw).hexdigest()
    ids = [x.strip() for x in cohort_raw.decode("utf-8-sig").splitlines() if x.strip()]
    require(len(ids) == HOMES and len(set(ids)) == HOMES,
            "Cohort ID count/uniqueness mismatch")
    require(cohort_manifest.get("household_ids_sha256") == cohort_sha,
            "Step 22 manifest / IDs checksum mismatch")
    require(summary.get("cohort_sha256") == cohort_sha,
            "Gold summary / cohort ID checksum mismatch")
    require(summary.get("cohort_size") == HOMES,
            "Gold cohort size mismatch")
    require(summary.get("cohort_selected_using_only_2013H1") is True,
            "Train-only cohort selection confirmation missing")
    require(cohort_manifest.get("selection_uses_future_validation_or_test_data") is False,
            "Step 22 is not marked as train-only selection")
    require(summary.get("version") == VERSION, "Unexpected Step 24 output version")
    require(summary.get("naive_timestamp_no_timezone_conversion") is True,
            "Timestamp semantics differ from Step 24")
    require(summary.get("hours") == HOURS and summary.get("source_count") == SOURCE_COUNT,
            "Gold summary source/hour count mismatch")
    require(summary.get("expected_readings_per_hour") == EXPECTED_PER_HOUR,
            "Gold summary expected readings/hour mismatch")

    print("\n1/4 Checking Gold Parquet rows and timestamp grid...")
    df = pd.read_parquet(GOLD)
    require(list(df.columns) == GOLD_FIELDS, "Gold Parquet column names/order mismatch")
    require(len(df) == HOURS, f"Expected {HOURS} hourly rows, got {len(df)}")
    hours = pd.to_datetime(df["hour_start"], errors="coerce").astype("datetime64[ns]")
    require(not hours.isna().any(), "Invalid Gold timestamps")
    require(hours.dt.tz is None, "Expected timezone-naive hour_start")
    target_hours = pd.date_range(START, periods=HOURS, freq="h")
    require(pd.DatetimeIndex(hours).equals(target_hours),
            "Gold hours must be sorted, unique, contiguous, with no missing hour")
    require(hours.iloc[0].to_pydatetime() == START and
            hours.iloc[-1].to_pydatetime() == END - timedelta(hours=1),
            "Gold timestamps have wrong date boundaries")

    # Exact integer checks, separate from floating-point sums.
    for name in ("observed_readings", "expected_readings", "missing_readings"):
        require(not df[name].isna().any(), f"NULL values in {name}")
        require(pd.api.types.is_integer_dtype(df[name]),
                f"Expected integer dtype in {name}")
    observed = df["observed_readings"].to_numpy(dtype="int64")
    expected = df["expected_readings"].to_numpy(dtype="int64")
    missing = df["missing_readings"].to_numpy(dtype="int64")
    require(np.all(expected == EXPECTED_PER_HOUR), "Expected readings vary across hours")
    require(np.all((observed >= 0) & (observed <= expected)),
            "Hourly observed reading count is outside [0, 10208]")
    require(np.array_equal(missing, expected - observed),
            "Missing reading counts do not match expected minus observed")
    require(pd.api.types.is_bool_dtype(df["is_complete"]),
            "is_complete must have boolean dtype")
    is_complete = df["is_complete"].to_numpy(dtype=bool)
    require(np.array_equal(is_complete, observed == EXPECTED_PER_HOUR),
            "is_complete does not match observed reading count")

    coverage = df["coverage_pct"].to_numpy(dtype="float64")
    target_coverage = np.round(100.0 * observed / expected, 6)
    require(np.all(np.isfinite(coverage)), "Non-finite coverage values")
    require(close_values(coverage, target_coverage, atol=6e-7, rtol=0),
            "Hourly coverage_pct differs from readings-based coverage")
    obs_kwh = df["observed_total_kwh"].to_numpy(dtype="float64")
    full_kwh = df["complete_total_kwh"].to_numpy(dtype="float64")
    require(np.array_equal(np.isnan(obs_kwh), observed == 0),
            "observed_total_kwh must be NULL exactly when no readings exist")
    require(np.all(np.isfinite(obs_kwh[observed > 0])) and
            np.all(obs_kwh[observed > 0] >= 0),
            "Non-finite/negative observed hourly kWh")
    require(np.array_equal(np.isnan(full_kwh), ~is_complete),
            "complete_total_kwh MUST be NULL for every incomplete hour")
    require(close_values(full_kwh[is_complete], obs_kwh[is_complete]),
            "Complete hourly total differs from observed total")

    total_observed = int(observed.sum())
    total_missing = int(missing.sum())
    complete_hours = int(is_complete.sum())
    require(total_observed + total_missing == HOURS * EXPECTED_PER_HOUR,
            "Yearly observed + missing != total expected half-hour slots")
    summary_counts = {
        "observed_cohort_readings": total_observed,
        "missing_cohort_half_hour_readings": total_missing,
        "completely_observed_hours": complete_hours,
        "incomplete_hours": HOURS - complete_hours,
        "hours_with_any_readings": int((observed > 0).sum()),
    }
    for key, value in summary_counts.items():
        require(summary.get(key) == value,
                f"Step 24 summary mismatch in {key}: {summary.get(key)} != {value}")

    print("2/4 Matching Gold Parquet against hourly CSV profile...")
    profile = pd.read_csv(PROFILE, encoding="utf-8-sig", parse_dates=["hour_start"])
    require(list(profile.columns) == GOLD_FIELDS, "Hourly profile fieldnames mismatch")
    require(len(profile) == HOURS, "Hourly CSV has an unexpected number of rows")
    require(pd.DatetimeIndex(profile["hour_start"]).equals(target_hours),
            "Hourly profile timestamp grid differs from Gold")
    for name in ("observed_readings", "expected_readings", "missing_readings"):
        require(np.array_equal(profile[name].to_numpy(dtype="int64"),
                               df[name].to_numpy(dtype="int64")),
                f"Hourly profile mismatch in {name}")
    parsed_bool = profile["is_complete"].map(
        lambda x: x if isinstance(x, bool) else str(x).strip().lower() == "true"
    ).to_numpy(dtype=bool)
    require(np.array_equal(parsed_bool, is_complete), "CSV/Gold completeness mismatch")
    for name in ("coverage_pct", "observed_total_kwh", "complete_total_kwh"):
        require(close_values(profile[name], df[name]),
                f"Hourly profile and Gold disagree on {name}")

    print("3/4 Reconciling all 168 per-file source-hour checkpoints...")
    require(PARTIAL.is_dir(), "Step 24 hourly source partial directory missing")
    meta_files = list(PARTIAL.glob("*.json"))
    csv_files = list(PARTIAL.glob("*.csv"))
    require(len(meta_files) == SOURCE_COUNT and len(csv_files) == SOURCE_COUNT,
            "Expected 168 source CSV+JSON checkpoint pairs")
    global_counts = np.zeros(HOURS, dtype="int64")
    global_kwh = np.zeros(HOURS, dtype="float64")
    time_index = {ts.strftime("%Y-%m-%d %H:%M:%S"): i
                  for i, ts in enumerate(target_hours)}
    partial_total = 0
    for meta_path in sorted(meta_files):
        meta = read_json(meta_path)
        part_path = PARTIAL / (meta_path.stem + ".csv")
        require(part_path.is_file(), f"Missing partial CSV for {meta_path.stem}")
        require(meta.get("source") == meta_path.stem, f"Wrong source in {meta_path.name}")
        require(meta.get("version") == VERSION, f"Wrong version in {meta_path.name}")
        require(meta.get("cohort_sha256") == cohort_sha,
                f"Cohort checksum mismatch in {meta_path.name}")
        require(meta.get("window") == "2013-01-01/2014-01-01",
                f"Wrong date window in {meta_path.name}")
        require(meta.get("csv_sha256") == sha_file(part_path),
                f"Partial CSV SHA-256 mismatch in {meta_path.name}")
        last_idx = -1
        group_count = 0
        source_rows = 0
        with part_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            require(reader.fieldnames == PARTIAL_FIELDS,
                    f"Partial fieldnames mismatch: {part_path.name}")
            for record in reader:
                stamp = record["hour_start"]
                require(stamp in time_index, f"Unexpected source hour: {stamp}")
                idx = time_index[stamp]
                require(idx > last_idx,
                        f"Unordered/duplicate source hour in {part_path.name}: {stamp}")
                n = int(record["observed_readings"])
                kwh = float(record["observed_kwh"])
                require(0 < n <= EXPECTED_PER_HOUR,
                        f"Invalid reading count in {part_path.name}: {stamp}")
                require(math.isfinite(kwh) and kwh >= 0,
                        f"Invalid kWh in {part_path.name}: {stamp}")
                global_counts[idx] += n
                global_kwh[idx] += kwh
                source_rows += n
                group_count += 1
                last_idx = idx
        require(meta.get("cohort_readings") == source_rows and
                meta.get("hour_groups") == group_count,
                f"Checkpoint row/group totals differ from CSV in {meta_path.name}")
        partial_total += source_rows
    require(partial_total == total_observed, "Sum of source readings differs from Gold")
    require(np.array_equal(global_counts, observed),
            "Source hourly reading counts do not match Gold counts")
    require(close_values(np.where(global_counts > 0, global_kwh, np.nan), obs_kwh),
            "Reconstructed source hourly kWh does not match Gold kWh")

    print("4/4 Writing validation & monthly quality diagnostics...")
    report = pd.DataFrame({
        "month": hours.dt.strftime("%Y-%m"),
        "observed_readings": observed,
        "missing_readings": missing,
        "expected_readings": expected,
        "complete_hour": is_complete.astype("int64"),
    })
    monthly = (report.groupby("month", sort=True, as_index=False)
               .agg(hours=("complete_hour", "size"),
                    complete_hours=("complete_hour", "sum"),
                    observed_readings=("observed_readings", "sum"),
                    missing_readings=("missing_readings", "sum"),
                    expected_readings=("expected_readings", "sum")))
    monthly["reading_coverage_pct"] = (100.0 * monthly["observed_readings"]
                                        / monthly["expected_readings"]).round(6)
    monthly["complete_hours_pct"] = (100.0 * monthly["complete_hours"]
                                      / monthly["hours"]).round(6)
    validation = {
        "status": "PASSED",
        "validation_scope": "Gold parquet, CSV, cohort manifest, 168 source-hour partials/checkpoints",
        "cohort_households": HOMES,
        "cohort_sha256": cohort_sha,
        "hours": HOURS,
        "half_hour_readings_observed": total_observed,
        "half_hour_readings_expected": HOURS * EXPECTED_PER_HOUR,
        "half_hour_readings_missing": total_missing,
        "half_hour_reading_coverage_pct": round(100.0 * total_observed /
                                                (HOURS * EXPECTED_PER_HOUR), 6),
        "complete_hours": complete_hours,
        "incomplete_hours": HOURS - complete_hours,
        "complete_hours_pct": round(100.0 * complete_hours / HOURS, 6),
        "hourly_coverage_min_pct": round(float(np.min(coverage)), 6),
        "hourly_coverage_median_pct": round(float(np.median(coverage)), 6),
        "hours_at_least_99_0_pct_readings": int((coverage >= 99.0).sum()),
        "hours_at_least_99_5_pct_readings": int((coverage >= 99.5).sum()),
        "hours_at_least_99_9_pct_readings": int((coverage >= 99.9).sum()),
        "source_timezone_verified": False,
        "cohort_status": "PROVISIONAL",
        "caution": ("An incomplete hour's observed_total_kwh is only the sum of present readings; "
                    "do not treat it as the full-cohort total or fill missing with zero."),
    }
    # Write only after all validations pass.
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT_MONTHLY, index=False, encoding="utf-8-sig")
    tmp = OUT_JSON.with_name(OUT_JSON.name + ".tmp")
    tmp.write_text(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True)
                   + "\n", encoding="utf-8")
    tmp.replace(OUT_JSON)

    print("\nGOLD FINAL VALIDATION: PASSED")
    print("Households:", f"{HOMES:,}")
    print("Hours:", f"{HOURS:,}")
    print("Observed half-hour readings:", f"{total_observed:,}")
    print("Expected half-hour readings:", f"{HOURS * EXPECTED_PER_HOUR:,}")
    print("Missing half-hour readings:", f"{total_missing:,}")
    print("Overall reading coverage:",
          f"{validation['half_hour_reading_coverage_pct']:.4f}%")
    print("Fully complete hours:", f"{complete_hours:,}")
    print("Incomplete hours:", f"{HOURS - complete_hours:,}")
    print("Median hourly reading coverage:",
          f"{validation['hourly_coverage_median_pct']:.4f}%")
    print("Hours with coverage >=99.5%:",
          f"{validation['hours_at_least_99_5_pct_readings']:,}")
    print("Validation report:", OUT_JSON)
    print("Monthly quality report:", OUT_MONTHLY)
    print("CAUTION: DST/timezone remains unresolved; no fill or scaling applied.")


if __name__ == "__main__":
    main()
