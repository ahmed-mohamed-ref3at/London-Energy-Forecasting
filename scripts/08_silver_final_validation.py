"""Step 16: independent Silver manifest / Parquet footer validation.

Run from the project root:
    python scripts/16_silver_final_validation.py

Requires: pyarrow (already installed in the london-energy environment).
This is metadata + checkpoint validation; it does not scan every value in Parquet.
"""

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq


BASE = Path(__file__).resolve().parents[1]
SILVER_DIR = BASE / "data" / "silver" / "readings_by_file"
RAW_AUDIT = BASE / "reports" / "raw_audit_batched.csv"
DUPLICATE_DIR = BASE / "reports" / "global_duplicate_buckets"
REPORT_CSV = BASE / "reports" / "silver_final_validation.csv"
REPORT_JSON = BASE / "reports" / "silver_final_summary.json"

FLAG_NAMES = [
    "missing_household_id",
    "missing_timestamp",
    "invalid_timestamp",
    "missing_kwh",
    "invalid_kwh",
    "negative_kwh",
]

EXPECTED_VALID_COLUMNS = {
    "source_file", "household_id", "tariff_group",
    "timestamp_text", "reading_timestamp", "kwh",
}

EXPECTED_QUARANTINE_COLUMNS = {
    "source_file", "household_id", "timestamp_text",
    "record_status", *[f"is_{name}" for name in FLAG_NAMES],
}


def parquet_footer_count(folder, expected_columns, reference_schemas, kind):
    """Return row count from Parquet footers; validate schema consistency."""
    if not (folder / "_SUCCESS").is_file():
        raise ValueError(f"Missing _SUCCESS: {folder}")

    parts = sorted(folder.rglob("*.parquet"))
    count = 0
    for part in parts:
        reader = pq.ParquetFile(str(part))
        schema = reader.schema_arrow
        if not expected_columns.issubset(set(schema.names)):
            raise ValueError(f"Missing {kind} columns: {part}")
        if kind not in reference_schemas:
            reference_schemas[kind] = schema
        elif not schema.equals(reference_schemas[kind], check_metadata=False):
            raise ValueError(f"Inconsistent {kind} schema: {part}")
        count += reader.metadata.num_rows
    return count, len(parts)


if not SILVER_DIR.is_dir():
    raise FileNotFoundError(SILVER_DIR)

with RAW_AUDIT.open("r", newline="", encoding="utf-8-sig") as handle:
    audit = {}
    for item in csv.DictReader(handle):
        source = item["source_file"]
        if source in audit:
            raise ValueError(f"Duplicate Raw Audit entry: {source}")
        audit[source] = {
            "row_count": int(item["row_count"]),
            **{flag: int(item[flag]) for flag in FLAG_NAMES},
        }

if len(audit) != 168:
    raise ValueError(f"Expected 168 Raw Audit entries, got {len(audit)}")

expected_folders = {Path(name).stem for name in audit}
actual_folders = {p.name for p in SILVER_DIR.iterdir() if p.is_dir()}
if actual_folders != expected_folders:
    raise ValueError(
        f"Missing Silver folders: {sorted(expected_folders - actual_folders)}\n"
        f"Unexpected Silver folders: {sorted(actual_folders - expected_folders)}"
    )

# Verify the existing global duplicate audit is complete and consistent.
dup_reports = []
for bucket_id in range(64):
    path = DUPLICATE_DIR / f"bucket_{bucket_id:02d}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("bucket_id") != bucket_id or result.get("num_buckets") != 64:
        raise ValueError(f"Invalid duplicate checkpoint: {path}")
    dup_reports.append(result)

expected_duplicates = sum(int(x["extra_exact_rows"]) for x in dup_reports)
if sum(int(x["rows_in_bucket"]) for x in dup_reports) != 167_932_474:
    raise ValueError("Global duplicate audit row total mismatch")
if sum(int(x["cross_file_keys"]) for x in dup_reports) != 0:
    raise ValueError("Cross-file duplicate keys require review")
if sum(int(x["conflicting_keys"]) for x in dup_reports) != 0:
    raise ValueError("Conflicting duplicate keys require review")

reference_schemas = {}
records = []
reasons_total = {}
flags_total = {name: 0 for name in FLAG_NAMES}

for filename in sorted(audit, key=lambda s: int(Path(s).stem.rsplit("_", 1)[-1])):
    folder = SILVER_DIR / Path(filename).stem
    marker = folder / "_silver_complete.json"
    if not marker.is_file():
        raise ValueError(f"Missing Silver checkpoint: {filename}")
    checkpoint = json.loads(marker.read_text(encoding="utf-8"))
    expected = audit[filename]

    if checkpoint.get("source_file") != filename:
        raise ValueError(f"Source name mismatch: {filename}")
    if int(checkpoint["bronze_rows"]) != expected["row_count"]:
        raise ValueError(f"Bronze count mismatch: {filename}")

    valid_rows, valid_parts = parquet_footer_count(
        folder / "valid", EXPECTED_VALID_COLUMNS, reference_schemas, "valid"
    )
    quarantine_rows, quarantine_parts = parquet_footer_count(
        folder / "quarantine", EXPECTED_QUARANTINE_COLUMNS,
        reference_schemas, "quarantine"
    )

    if valid_rows != int(checkpoint["valid_rows"]):
        raise ValueError(f"Silver Valid footer mismatch: {filename}")
    if quarantine_rows != int(checkpoint["quarantine_rows"]):
        raise ValueError(f"Quarantine footer mismatch: {filename}")
    if valid_rows + quarantine_rows != expected["row_count"]:
        raise ValueError(f"Row reconciliation failed: {filename}")

    reason_counts = checkpoint["quarantine_reasons"]
    if sum(int(v) for v in reason_counts.values()) != quarantine_rows:
        raise ValueError(f"Quarantine reasons mismatch: {filename}")
    if int(reason_counts.get("duplicate_key", 0)) != int(checkpoint["duplicate_rows"]):
        raise ValueError(f"Duplicate reasons mismatch: {filename}")

    for flag in FLAG_NAMES:
        reported = int(checkpoint["quality_flags"][flag])
        if reported != expected[flag]:
            raise ValueError(f"Raw-vs-Quarantine flag mismatch: {filename}, {flag}")
        flags_total[flag] += reported

    for reason, amount in reason_counts.items():
        reasons_total[reason] = reasons_total.get(reason, 0) + int(amount)

    records.append({
        "source_file": filename,
        "bronze_rows": expected["row_count"],
        "valid_rows": valid_rows,
        "quarantine_rows": quarantine_rows,
        "duplicate_rows": int(checkpoint["duplicate_rows"]),
        "valid_parquet_parts": valid_parts,
        "quarantine_parquet_parts": quarantine_parts,
        "status": "PASSED",
    })

summary = {
    "files_validated": len(records),
    "bronze_rows": sum(r["bronze_rows"] for r in records),
    "valid_rows": sum(r["valid_rows"] for r in records),
    "quarantine_rows": sum(r["quarantine_rows"] for r in records),
    "duplicate_rows": sum(r["duplicate_rows"] for r in records),
    "quarantine_reasons": reasons_total,
    "quality_flags_in_quarantine": flags_total,
    "validation_method": "Parquet footers + checkpoint and audit reconciliation",
}

if summary["files_validated"] != 168:
    raise ValueError("Missing Silver sources")
if summary["bronze_rows"] != 167_932_474:
    raise ValueError("Unexpected global Bronze row count")
if summary["valid_rows"] != 167_811_461:
    raise ValueError("Unexpected global Silver Valid count")
if summary["quarantine_rows"] != 121_013:
    raise ValueError("Unexpected global Quarantine count")
if summary["duplicate_rows"] != expected_duplicates or expected_duplicates != 115_453:
    raise ValueError("Global duplicate reconciliation failed")
if flags_total["missing_kwh"] != 5_560:
    raise ValueError("Missing kWh reconciliation failed")
if reasons_total.get("missing_kwh", 0) != 5_560:
    raise ValueError("Missing kWh reason reconciliation failed")
if set(reasons_total) - {"duplicate_key", "missing_kwh"}:
    raise ValueError(f"Unexpected quarantine reasons: {reasons_total}")
if summary["valid_rows"] + summary["quarantine_rows"] != summary["bronze_rows"]:
    raise ValueError("Global row reconciliation failed")

REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
with REPORT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("=" * 55)
print("LONDON SMART METER - SILVER FINAL VALIDATION")
print("=" * 55)
print("Validated files:", summary["files_validated"])
for key in ["bronze_rows", "valid_rows", "quarantine_rows", "duplicate_rows"]:
    print(f"{key}: {summary[key]:,}")
print("missing_kwh_rows:", f"{reasons_total.get('missing_kwh', 0):,}")
print("Difference:", summary["bronze_rows"] - summary["valid_rows"] - summary["quarantine_rows"])
print("Schema consistency: PASSED")
print("Parquet footer counts: PASSED")
print("Checkpoint and audit reconciliation: PASSED")
print("\nSILVER FINAL VALIDATION: PASSED!")
print("File report:", REPORT_CSV)
print("Summary:", REPORT_JSON)
