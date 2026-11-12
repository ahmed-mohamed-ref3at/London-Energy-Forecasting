"""Step 22 - Freeze a PROVISIONAL training-only household cohort.

Input: reports/training_2013H1_household_gaps.csv from Step 21.
No future (Jul-Dec 2013) data are used to select households.

Pure Python standard library. No Spark or full data scan required.
Re-runs verify existing output bytes; they never silently overwrite a cohort.
The cohort is provisional pending verification of original time-zone/DST meaning.

Run: python scripts/22_freeze_training_cohort.py
"""

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
INPUT_CSV = BASE / "reports" / "training_2013H1_household_gaps.csv"
INPUT_SUMMARY = BASE / "reports" / "training_2013H1_gap_overview.json"
OUT = BASE / "reports" / "cohorts" / "2013H1_candidate_v1"

EXPECTED_HOUSEHOLDS = 5_561
EXPECTED_ELIGIBLE = 5_104
EXPECTED_TRAINING_ROWS = 47_017_289
EXPECTED_SLOTS = 8_688
MIN_COVERAGE_PCT = 99.5
MAX_GAP_HOURS = 6.0
TRAIN_START = "2013-01-01 00:00:00"
TRAIN_END_EXCLUSIVE = "2013-07-01 00:00:00"

REQUIRED = [
    "household_id", "observed_train_slots", "expected_train_slots",
    "missing_train_slots", "train_coverage_pct", "gap_runs",
    "longest_gap_slots", "longest_gap_hours",
    "first_slot_observed", "last_slot_observed",
]


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def stable_write(path, data):
    """Write once; on subsequent runs require exactly the same bytes."""
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(
                f"Frozen output has changed: {path}\n"
                "Do not overwrite an existing cohort. Review and use a new version."
            )
        print(f"VERIFIED existing: {path.name}")
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Leftover temporary file: {temporary}; inspect it first")
    temporary.write_bytes(data)
    temporary.replace(path)
    print(f"CREATED: {path.name}")


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: row[k] for k in fields} for row in rows)
    return stream.getvalue().encode("utf-8-sig")


if not INPUT_CSV.is_file() or not INPUT_SUMMARY.is_file():
    raise FileNotFoundError("Step 21 reports are missing. Complete Step 21 first.")

raw_bytes = INPUT_CSV.read_bytes()
with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    absent = set(REQUIRED) - set(reader.fieldnames or [])
    if absent:
        raise ValueError(f"Missing CSV columns: {sorted(absent)}")
    rows = list(reader)

summary = json.loads(INPUT_SUMMARY.read_text(encoding="utf-8"))
if summary.get("households_in_silver") != EXPECTED_HOUSEHOLDS:
    raise ValueError("Step 21 household count mismatch")
if summary.get("silver_valid_rows_in_training_window") != EXPECTED_TRAINING_ROWS:
    raise ValueError("Step 21 training row count mismatch")
if summary.get("expected_slots_per_household") != EXPECTED_SLOTS:
    raise ValueError("Step 21 expected slot count mismatch")
if summary.get("window_start_inclusive") != TRAIN_START:
    raise ValueError("Training start mismatch")
if summary.get("window_end_exclusive") != TRAIN_END_EXCLUSIVE:
    raise ValueError("Training end mismatch")
if summary.get("future_validation_included_in_selection") is not False:
    raise ValueError("Source summary does not guarantee train-only selection")

if len(rows) != EXPECTED_HOUSEHOLDS:
    raise ValueError(f"Expected {EXPECTED_HOUSEHOLDS} households, got {len(rows)}")

seen = set()
selected = []
audit = []
fail_reasons = Counter()
sum_observed = 0

for row in rows:
    hid = row["household_id"].strip()
    if not hid or hid in seen:
        raise ValueError(f"Missing or duplicate household_id: {hid!r}")
    seen.add(hid)
    observed = int(row["observed_train_slots"])
    expected = int(row["expected_train_slots"])
    missing = int(row["missing_train_slots"])
    coverage = float(row["train_coverage_pct"])
    gap_runs = int(row["gap_runs"])
    longest_slots = int(row["longest_gap_slots"])
    longest_hours = float(row["longest_gap_hours"])
    first = int(row["first_slot_observed"])
    last = int(row["last_slot_observed"])

    if (expected != EXPECTED_SLOTS or not (0 <= observed <= expected)
            or missing != expected - observed
            or abs(coverage - round(100 * observed / expected, 4)) > 0.00011
            or gap_runs < 0 or not (0 <= longest_slots <= missing)
            or longest_hours != longest_slots / 2
            or first not in (0, 1) or last not in (0, 1)
            or (missing == 0 and (gap_runs != 0 or longest_slots != 0))):
        raise ValueError(f"Training profile inconsistency for {hid}")
    sum_observed += observed

    reasons = []
    if coverage < MIN_COVERAGE_PCT:
        reasons.append("coverage_below_99_5pct")
    if longest_hours > MAX_GAP_HOURS:
        reasons.append("longest_gap_over_6h")
    if first != 1:
        reasons.append("first_training_slot_missing")
    if last != 1:
        reasons.append("last_training_slot_missing")

    eligible = not reasons
    entry = dict(row)
    entry["household_id"] = hid
    entry["eligible"] = "1" if eligible else "0"
    entry["exclusion_reasons"] = ";".join(reasons)
    audit.append(entry)

    if eligible:
        selected.append(dict(row, household_id=hid))
    else:
        fail_reasons.update(reasons)

if sum_observed != EXPECTED_TRAINING_ROWS:
    raise ValueError("Observed readings do not reconcile to Step 21")

selected.sort(key=lambda r: r["household_id"])
audit.sort(key=lambda r: r["household_id"])
if len(selected) != EXPECTED_ELIGIBLE:
    raise ValueError(
        f"Expected {EXPECTED_ELIGIBLE} candidates, got {len(selected)}. "
        "Do not save a different cohort before reviewing the criteria."
    )
if summary.get("preliminary_99_5pct_max_6h_both_endpoints_count") != len(selected):
    raise ValueError("Eligible count differs from Step 21 summary")

OUT.mkdir(parents=True, exist_ok=True)
cohort_data = csv_bytes(selected, REQUIRED)
all_audit_data = csv_bytes(audit, REQUIRED + ["eligible", "exclusion_reasons"])
ids_data = ("\n".join(r["household_id"] for r in selected) + "\n").encode("utf-8")

manifest = {
    "cohort_id": "2013H1_candidate_v1",
    "status": "PROVISIONAL - timestamp timezone/DST meaning not verified",
    "time_basis": "naive half-hour TIMESTAMP_NTZ",
    "training_start_inclusive": TRAIN_START,
    "training_end_exclusive": TRAIN_END_EXCLUSIVE,
    "selection_uses_future_validation_or_test_data": False,
    "eligibility": {
        "min_training_coverage_pct": MIN_COVERAGE_PCT,
        "max_consecutive_gap_hours": MAX_GAP_HOURS,
        "first_training_slot_required": True,
        "last_training_slot_required": True,
    },
    "households_audited": len(rows),
    "households_eligible": len(selected),
    "households_excluded": len(rows) - len(selected),
    "exclusion_reason_counts_overlapping": dict(sorted(fail_reasons.items())),
    "training_rows_reconciled": sum_observed,
    "source_report": str(INPUT_CSV.relative_to(BASE)).replace("\\", "/"),
    "source_report_sha256": sha256_bytes(raw_bytes),
    "cohort_csv_sha256": sha256_bytes(cohort_data),
    "household_ids_sha256": sha256_bytes(ids_data),
    "note": "Frozen candidate selection, not a verification of holdout coverage. "
            "Do not reselect using Jul-Dec 2013 observations. "
            "Resolve time semantics and target missing-data policy before final Gold.",
}
manifest_data = (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")

# All files are immutable by default. If changed input leads to changed outputs,
# abort rather than silently modifying a previously adopted cohort version.
for filename, data in [
    ("eligible_households.csv", cohort_data),
    ("household_ids.txt", ids_data),
    ("selection_audit.csv", all_audit_data),
    ("manifest.json", manifest_data),
]:
    stable_write(OUT / filename, data)

print("=" * 60)
print("STEP 22 - TRAINING-ONLY COHORT CANDIDATE")
print("=" * 60)
print("Households audited:", f"{len(rows):,}")
print("Eligible household IDs:", f"{len(selected):,}")
print("Excluded households:", f"{len(rows) - len(selected):,}")
print("Train-only selection: VERIFIED")
print("Training rows reconciled:", f"{sum_observed:,}")
print("Selection rule: coverage >=99.5%, longest gap <=6h, both endpoints")
print("Cohort status: PROVISIONAL (DST/timezone unresolved)")
print("COHORT CANDIDATE FREEZE: PASSED!")
print("Output directory:", OUT)
