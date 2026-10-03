"""Convert the raw ODS files to Parquet, one file per financial year sheet.

All columns are written as strings, verbatim. No recoding, type conversion or
cleaning is applied at this stage.

The column header of each sheet is asserted against fdh.contract and a
mismatch stops the run. Incident type and day of week values outside the
contract are counted, reported, and written through unchanged.

Also writes conversion_record.json: row counts and value counts per sheet.

Usage:
    uv run python scripts/convert_raw.py
    uv run python scripts/convert_raw.py --only 2023-24_to_present
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdh.config import load_config  # noqa: E402
from fdh.contract import (  # noqa: E402
    DATA_SHEET_PATTERN,
    EXPECTED_COLUMNS,
    KNOWN_DAY_OF_WEEK,
    KNOWN_INCIDENT_LABELS,
)
from fdh.ods import OdsStructureError, iter_rows, sheet_names  # noqa: E402

BATCH_ROWS = 200_000


def convert_sheet(path: Path, sheet: str, out_path: Path) -> dict:
    header: list[str] | None = None
    batch: list[list[str]] = []
    writer: pq.ParquetWriter | None = None
    schema = pa.schema([(name, pa.string()) for name in EXPECTED_COLUMNS])
    rows = 0
    labels: Counter[str] = Counter()
    days: Counter[str] = Counter()
    ragged = 0

    def flush() -> None:
        nonlocal batch, writer
        if not batch:
            return
        columns = list(zip(*batch, strict=True))
        table = pa.Table.from_arrays(
            [pa.array(column, type=pa.string()) for column in columns], schema=schema
        )
        writer.write_table(table)
        batch = []

    for _, cells in iter_rows(path, sheet=sheet):
        if header is None:
            header = [cell.strip() for cell in cells]
            if tuple(header) != EXPECTED_COLUMNS:
                raise OdsStructureError(
                    f"{path.name}, sheet {sheet}: unexpected columns.\n"
                    f"  expected: {EXPECTED_COLUMNS}\n"
                    f"  found:    {tuple(header)}"
                )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(out_path, schema, compression="zstd")
            continue

        if len(cells) != len(EXPECTED_COLUMNS):
            ragged += 1
            if ragged <= 5:
                print(
                    f"  ragged row {rows + 1} in {sheet}: {len(cells)} fields, {cells}",
                    file=sys.stderr,
                )
            if len(cells) < len(EXPECTED_COLUMNS):
                cells = cells + [""] * (len(EXPECTED_COLUMNS) - len(cells))
            else:
                raise OdsStructureError(
                    f"{path.name}, sheet {sheet}: row {rows + 1} has {len(cells)} fields, "
                    f"more than the {len(EXPECTED_COLUMNS)} columns published"
                )

        batch.append(cells)
        rows += 1
        labels[cells[8]] += 1
        days[cells[7]] += 1
        if len(batch) >= BATCH_ROWS:
            flush()

    if header is None:
        raise OdsStructureError(f"{path.name}, sheet {sheet}: no rows found")
    flush()
    if writer is not None:
        writer.close()

    unknown_labels = sorted(set(labels) - KNOWN_INCIDENT_LABELS)
    unknown_days = sorted(set(days) - KNOWN_DAY_OF_WEEK)

    return {
        "sheet": sheet,
        "rows": rows,
        "ragged_rows": ragged,
        "parquet": str(out_path),
        "incident_type_counts": dict(labels.most_common()),
        "day_of_week_counts": dict(sorted(days.items())),
        "unknown_incident_labels": unknown_labels,
        "unknown_day_of_week": unknown_days,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="convert a single era by its config key")
    args = parser.parse_args()

    config = load_config()
    raw_dir = config.paths.resolved("raw_dir")
    interim_dir = config.paths.resolved("interim_dir")

    records = []
    novel: list[str] = []

    for sources in config.sources.values():
        for source in sources:
            if args.only and source.era != args.only:
                continue
            path = raw_dir / source.filename
            if not path.exists():
                raise SystemExit(f"missing raw file: {path}")

            print(f"\n{source.filename}")
            present = sheet_names(path)
            data_sheets = [name for name in present if DATA_SHEET_PATTERN.match(name)]
            if not data_sheets:
                raise OdsStructureError(
                    f"{source.filename}: no sheet matching a six digit financial year. "
                    f"Sheets present: {present}"
                )
            print(f"  sheets: {present}")

            for sheet in data_sheets:
                out_path = interim_dir / source.era / f"{sheet}.parquet"
                record = convert_sheet(path, sheet, out_path)
                record["era"] = source.era
                record["source_file"] = source.filename
                records.append(record)
                print(f"  {sheet}: {record['rows']:,} rows -> {out_path.name}")
                if record["unknown_incident_labels"]:
                    novel.append(
                        f"{source.era}/{sheet}: incident labels not in the "
                        f"2023-24 contract: {record['unknown_incident_labels']}"
                    )
                if record["unknown_day_of_week"]:
                    novel.append(
                        f"{source.era}/{sheet}: day of week values not in the "
                        f"contract: {record['unknown_day_of_week']}"
                    )

    interim_dir.mkdir(parents=True, exist_ok=True)
    record_path = interim_dir / "conversion_record.json"
    record_path.write_text(
        json.dumps(
            {
                "converted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "sheets": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {record_path}")
    print(f"total rows: {sum(record['rows'] for record in records):,}")

    if novel:
        print("\nValues outside the 2023-24 contract:")
        for line in novel:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())