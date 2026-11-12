
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parents[1]

BRONZE_DIR = (
    BASE_DIR / "data" / "bronze" / "readings_by_file"
)

AUDIT_PATH = BASE_DIR / "reports" / "raw_audit_batched.csv"

REPORT_PATH = BASE_DIR / "reports" / "bronze_validation.csv"

METRICS = [
    "row_count",
    "missing_household_id",
    "missing_timestamp",
    "invalid_timestamp",
    "missing_kwh",
    "invalid_kwh",
    "negative_kwh",
]


# 1. Load expected results from Raw Audit
with AUDIT_PATH.open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    raw_rows = list(csv.DictReader(file))

raw_audit = {}

for row in raw_rows:
    name = row["source_file"]

    if name in raw_audit:
        raise ValueError(f"Duplicate audit entry: {name}")

    raw_audit[name] = {
        metric: int(row[metric])
        for metric in METRICS
    }

if len(raw_audit) != 168:
    raise ValueError("Expected 168 raw audit records")


# 2. Verify all expected Bronze directories
expected_dirs = {
    Path(name).stem
    for name in raw_audit
}

actual_dirs = {
    p.name
    for p in BRONZE_DIR.iterdir()
    if p.is_dir()
}

if actual_dirs != expected_dirs:
    raise ValueError(
        f"Missing directories: {expected_dirs - actual_dirs}\n"
        f"Unexpected directories: {actual_dirs - expected_dirs}"
    )


# 3. Validate each Parquet dataset
results = []
reference_schema = None

for source_name in sorted(raw_audit):

    folder = BRONZE_DIR / Path(source_name).stem

    marker = folder / "_bronze_complete.json"
    success = folder / "_SUCCESS"

    if not marker.is_file() or not success.is_file():
        raise ValueError(
            f"Missing completion files: {folder.name}"
        )

    checkpoint = json.loads(
        marker.read_text(encoding="utf-8")
    )

    expected = raw_audit[source_name]

    if checkpoint.get("source_file") != source_name:
        raise ValueError(
            f"Source name mismatch: {source_name}"
        )

    if checkpoint.get("metrics") != expected:
        raise ValueError(
            f"Checkpoint mismatch: {source_name}"
        )

    parquet_parts = sorted(
        folder.rglob("*.parquet")
    )

    if not parquet_parts:
        raise ValueError(
            f"No Parquet files found: {source_name}"
        )

    parquet_count = 0

    for part in parquet_parts:

        parquet = pq.ParquetFile(str(part))

        parquet_count += parquet.metadata.num_rows

        schema = parquet.schema_arrow

        if reference_schema is None:
            reference_schema = schema
        elif not schema.equals(
            reference_schema,
            check_metadata=False
        ):
            raise ValueError(
                f"Schema mismatch in: {part}"
            )

    if parquet_count != expected["row_count"]:
        raise ValueError(
            f"Row mismatch in {source_name}: "
            f"Expected={expected['row_count']}, "
            f"Actual={parquet_count}"
        )

    results.append({
        "source_file": source_name,
        "raw_rows": expected["row_count"],
        "bronze_rows": parquet_count,
        "parquet_parts": len(parquet_parts),
        "status": "PASSED",
    })


# 4. Verify full dataset total
raw_total = sum(
    item["raw_rows"] for item in results
)

bronze_total = sum(
    item["bronze_rows"] for item in results
)

if raw_total != bronze_total:
    raise ValueError("Global row count mismatch!")

if raw_total != 167_932_474:
    raise ValueError(
        f"Unexpected dataset size: {raw_total}"
    )


# 5. Save final validation report
with REPORT_PATH.open(
    "w", newline="", encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file, fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


print("=" * 55)
print("LONDON SMART METER - BRONZE VALIDATION")
print("=" * 55)

print("Validated files:", len(results))
print("Raw rows:", f"{raw_total:,}")
print("Bronze rows:", f"{bronze_total:,}")
print("Difference:", bronze_total - raw_total)

print(
    "Missing kWh (previously validated):",
    sum(
        record["missing_kwh"]
        for record in raw_audit.values()
    )
)

print("Schema consistency: PASSED")
print("Checkpoints: PASSED")
print("Parquet metadata: PASSED")
print("\nFINAL BRONZE VALIDATION: PASSED")
print("Report:", REPORT_PATH)
