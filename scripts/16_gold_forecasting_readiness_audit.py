"""Step 27: read-only forecasting readiness audit for 2013 fixed-cohort Gold.

From the London-Energy-Forecasting project root, run:
    python scripts/27_gold_forecasting_readiness_audit.py

Requires: pandas, numpy, pyarrow (already used by the project's Step 25).
Reads Gold only; never changes it. Creates report files under reports/.
No imputation, feature engineering, model training or timezone conversion.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = [
    "hour_start", "observed_total_kwh", "complete_total_kwh",
    "observed_readings", "expected_readings", "missing_readings",
    "coverage_pct", "is_complete",
]
EXPECTED_HOURS = 8760
COHORT_SIZE = 5104
EXPECTED_READINGS_PER_HOUR = 2 * COHORT_SIZE


def ensure(condition, explanation):
    if not bool(condition):
        raise ValueError("GOLD VALIDATION FAILED: " + explanation)


def maximum_run(series, desired_value):
    longest = current = 0
    for value in series:
        current = current + 1 if bool(value) == desired_value else 0
        longest = max(longest, current)
    return longest


def audit_dataframe(data):
    """Validate the historical Gold schema and return a report plus small CSV tables."""
    df = data.copy()
    ensure(list(df.columns) == FIELDS, f"Unexpected Gold columns: {list(df.columns)}")
    ensure(len(df) == EXPECTED_HOURS, f"Expected 8760 hours; found {len(df)}")

    ts = pd.to_datetime(df["hour_start"], errors="coerce")
    ensure(ts.notna().all(), "Invalid hour_start timestamps")
    ensure(ts.dt.tz is None, "Expected recorded timezone-naive timestamps")
    expected_grid = pd.date_range("2013-01-01", periods=EXPECTED_HOURS, freq="h")
    # Parquet is an unordered storage format. Reading a directory of part files
    # does NOT guarantee chronological row order. Normalize time precision and
    # check the set of timestamps before applying time-dependent calculations.
    df = df.assign(hour_start=ts).sort_values("hour_start", kind="stable").reset_index(drop=True)
    ts = df["hour_start"]
    actual_ns = ts.to_numpy(dtype="datetime64[ns]")
    expected_ns = expected_grid.to_numpy(dtype="datetime64[ns]")
    if not np.array_equal(actual_ns, expected_ns):
        actual_index = pd.DatetimeIndex(actual_ns)
        missing_hours = expected_grid.difference(actual_index)
        unexpected_hours = actual_index.difference(expected_grid)
        duplicate_hours = ts[ts.duplicated(keep=False)]
        raise ValueError(
            "GOLD VALIDATION FAILED: 2013 hourly timestamp coverage differs "
            "from the expected grid AFTER sorting. "
            f"Earliest actual: {ts.iloc[0]}, latest actual: {ts.iloc[-1]}; "
            f"missing hours: {len(missing_hours)} "
            f"(examples: {list(missing_hours[:5])}); "
            f"unexpected hours: {len(unexpected_hours)} "
            f"(examples: {list(unexpected_hours[:5])}); "
            f"duplicate rows: {len(duplicate_hours)} "
            f"(examples: {list(duplicate_hours.head(5))}). "
            "Do not modify Gold; inspect this diagnostic first."
        )

    for col in ("observed_readings", "expected_readings", "missing_readings"):
        ensure(df[col].notna().all(), f"NULL counts in {col}")
        ensure(pd.api.types.is_integer_dtype(df[col]), f"{col} should be integer")

    observed = df["observed_readings"].to_numpy(dtype=np.int64)
    expected = df["expected_readings"].to_numpy(dtype=np.int64)
    missing = df["missing_readings"].to_numpy(dtype=np.int64)
    ensure(np.all(expected == EXPECTED_READINGS_PER_HOUR),
           "Expected count per hour differs from 10208")
    ensure(np.all((observed >= 0) & (observed <= expected)),
           "Observed reading counts outside valid range")
    ensure(np.array_equal(missing, expected - observed),
           "Missing reading counts are inconsistent")
    ensure(pd.api.types.is_bool_dtype(df["is_complete"]),
           "is_complete should be Boolean")
    complete = df["is_complete"].to_numpy(dtype=bool)
    ensure(np.array_equal(complete, missing == 0),
           "is_complete is inconsistent with missing_readings")

    coverage = df["coverage_pct"].to_numpy(dtype=float)
    ensure(np.isfinite(coverage).all(), "Invalid hourly coverage percentages")
    calculated_coverage = np.round(100.0 * observed / expected, 6)
    ensure(np.allclose(coverage, calculated_coverage, atol=6e-7, rtol=0),
           "Coverage percentages do not match observed/expected readings")

    observed_energy = df["observed_total_kwh"].to_numpy(dtype=float)
    full_energy = df["complete_total_kwh"].to_numpy(dtype=float)
    ensure(np.array_equal(np.isnan(observed_energy), observed == 0),
           "Observed energy NULL status is inconsistent with observed count")
    ensure(np.isfinite(observed_energy[observed > 0]).all(),
           "Nonfinite observed energy")
    ensure((observed_energy[observed > 0] >= 0).all(),
           "Negative observed energy")
    ensure(np.array_equal(np.isnan(full_energy), ~complete),
           "Complete energy must be NULL exactly when hour is incomplete")
    ensure(np.allclose(full_energy[complete], observed_energy[complete],
                       rtol=1e-10, atol=1e-6),
           "Complete-hour and observed-hour energy totals disagree")

    missing_bins = [
        ("0 (fully complete)", missing == 0),
        ("1-2", (missing >= 1) & (missing <= 2)),
        ("3-10", (missing >= 3) & (missing <= 10)),
        ("11-50", (missing >= 11) & (missing <= 50)),
        ("51-100", (missing >= 51) & (missing <= 100)),
        ("101-500", (missing >= 101) & (missing <= 500)),
        ("501-1000", (missing >= 501) & (missing <= 1000)),
        (">1000", missing > 1000),
    ]
    buckets = pd.DataFrame([
        {"missing_readings_range": name, "hours": int(mask.sum()),
         "hours_pct": round(float(mask.mean() * 100), 3)}
        for name, mask in missing_bins
    ])
    ensure(int(buckets["hours"].sum()) == EXPECTED_HOURS,
           "Missing-readings bucket reconciliation failed")

    summary = {
        "source": "data/gold/hourly_2013_cohort_v1",
        "purpose": "Read-only missingness audit, not a finalized modeling target",
        "gold_validation": "PASSED: timestamp grid, counts, flags, coverage and energy consistency",
        "cohort_households": COHORT_SIZE,
        "total_hours": int(len(df)),
        "fully_complete_hours": int(complete.sum()),
        "incomplete_hours": int((~complete).sum()),
        "fully_complete_hours_pct": round(float(complete.mean() * 100), 4),
        "expected_halfhour_readings": int(expected.sum()),
        "observed_halfhour_readings": int(observed.sum()),
        "missing_halfhour_readings": int(missing.sum()),
        "overall_reading_coverage_pct": round(float(100 * observed.sum() / expected.sum()), 6),
        "missing_readings_per_hour": {
            "mean": round(float(np.mean(missing)), 3),
            "median": float(np.median(missing)),
            "p90": float(np.percentile(missing, 90)),
            "p95": float(np.percentile(missing, 95)),
            "p99": float(np.percentile(missing, 99)),
            "max": int(missing.max()),
        },
        "hours_meeting_reading_coverage_thresholds": {
            str(threshold) + "%": int(np.count_nonzero(coverage >= threshold))
            for threshold in (100.0, 99.9, 99.5, 99.0, 95.0)
        },
        "longest_consecutive_fully_complete_hours": maximum_run(complete, True),
        "longest_consecutive_incomplete_hours": maximum_run(complete, False),
        "interpretation_note": (
            "Near-complete hours are NOT verified full-cohort consumption targets. "
            "Threshold counts are diagnostics only; no imputation or target policy was selected. "
            "The aggregate Gold file cannot reveal which household readings are missing."
        ),
    }

    monthly = pd.DataFrame({
        "month": ts.dt.strftime("%Y-%m"),
        "missing_readings": missing,
        "is_complete": complete,
    }).groupby("month", sort=True).agg(
        hours=("is_complete", "size"),
        fully_complete_hours=("is_complete", "sum"),
        total_missing_readings=("missing_readings", "sum"),
        median_missing_per_hour=("missing_readings", "median"),
        max_missing_per_hour=("missing_readings", "max"),
    ).reset_index()
    monthly["complete_hours_pct"] = (
        100 * monthly["fully_complete_hours"] / monthly["hours"]
    ).round(3)

    worst = pd.DataFrame({
        "hour_start": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "missing_readings": missing,
        "observed_readings": observed,
        "coverage_pct": coverage,
    }).sort_values(["missing_readings", "hour_start"], ascending=[False, True]).head(20)

    return summary, buckets, monthly, worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--gold", type=Path,
                        default=default_root / "data/gold/hourly_2013_cohort_v1")
    parser.add_argument("--out", type=Path,
                        default=default_root / "reports/forecasting_readiness_v1")
    args = parser.parse_args()

    ensure(args.gold.exists(), f"Gold path not found: {args.gold}")
    print("Reading 2013 hourly Gold (read-only):", args.gold)
    df = pd.read_parquet(args.gold)
    summary, buckets, monthly, worst = audit_dataframe(df)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gold_readiness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    buckets.to_csv(args.out / "missing_readings_distribution.csv", index=False)
    monthly.to_csv(args.out / "missingness_by_month.csv", index=False)
    worst.to_csv(args.out / "worst_20_hours.csv", index=False)

    print("\n" + "=" * 62)
    print("STEP 27 — GOLD FORECASTING READINESS AUDIT: PASSED")
    print("=" * 62)
    print("Total hours:", f"{summary['total_hours']:,}")
    print("Complete hours:", f"{summary['fully_complete_hours']:,}")
    print("Incomplete hours:", f"{summary['incomplete_hours']:,}")
    print("Overall reading coverage:", f"{summary['overall_reading_coverage_pct']:.4f}%")
    print("Missing readings per hour:", summary["missing_readings_per_hour"])
    print("\nHours by missing-count range:")
    print(buckets.to_string(index=False))
    print("\nLongest complete-hour streak:",
          summary["longest_consecutive_fully_complete_hours"])
    print("Longest incomplete-hour streak:",
          summary["longest_consecutive_incomplete_hours"])
    print("\nReports saved to:", args.out.resolve())
    print("No source files modified. No target-definition decision made.")


if __name__ == "__main__":
    main()
