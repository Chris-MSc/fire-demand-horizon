"""Build the dense daily incident panel from the converted files.

Writes one row per calendar day, territory and incident type across the span
set by panel.start_date and panel.end_date, falling back to the recorded span
for either end left unset, with zero where nothing was recorded and null where
the day falls inside a reporting gap declared in config.

The build stops, writing nothing, if a single row gives no real date or a date
that disagrees with the published weekday or financial year.

Also writes panel_record.json alongside the panel, and two reports:

    panel_zero_shares.csv   zero days per series, the basis for the minimum
                            non-zero day count used for scoring eligibility
    panel_quiet_days.csv    territory-days with no incidents of any type,
                            outside a declared reporting gap

Usage:
    uv run python scripts/build_panel.py
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fdh.config import incident_sources, load_config
from fdh.panel import build_rows, quiet_days, read_counts, zero_shares
from fdh.provenance import sha256_file

PANEL_SCHEMA = pa.schema(
    [
        ("incident_date", pa.date32()),
        ("e_code_territory", pa.string()),
        ("frs_territory", pa.string()),
        ("incident_type", pa.string()),
        ("count", pa.int32()),
        ("reporting_gap", pa.bool_()),
    ]
)

QUIET_COLUMNS = (
    "incident_date",
    "e_code_territory",
    "frs_territory",
    "incidents",
    "territory_median",
)


def write_panel(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [pa.array([row[field.name] for row in rows], type=field.type) for field in PANEL_SCHEMA],
        schema=PANEL_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")


def write_csv(rows: list[dict], columns: tuple[str, ...] | list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--types",
        nargs="*",
        default=None,
        help="Incident types to build. Defaults to incident_types.modelled_types.",
    )
    args = parser.parse_args()

    config = load_config()
    interim = config.paths.resolved("interim_dir")
    processed = config.paths.resolved("processed_dir")
    reports = config.paths.resolved("reports_dir")
    types = args.types or config.incident_types.modelled_types
    eras = [source.era for source in incident_sources(config)]
    gaps = [
        (gap.e_code_territory, gap.start, gap.end)
        for gap in config.validation.reporting_gaps
    ]
    first_day = config.panel.start_date
    last_day = config.panel.end_date

    print(f"Reading converted incidents from {interim}")
    result = read_counts(interim, eras)
    rows, record = build_rows(result, types, gaps=gaps, first_day=first_day, last_day=last_day)

    panel_path = processed / "daily_counts.parquet"
    write_panel(rows, panel_path)

    summary = zero_shares(rows)
    shares_path = reports / "panel_zero_shares.csv"
    write_csv(summary, list(summary[0]), shares_path)

    quiet = quiet_days(result, first_day or result.first_day, last_day or result.last_day, gaps)
    quiet_path = reports / "panel_quiet_days.csv"
    write_csv(quiet, QUIET_COLUMNS, quiet_path)

    conversion_record = interim / "conversion_record.json"
    record["built_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["panel_file"] = panel_path.name
    record["series"] = len(summary)
    record["days_with_no_incidents_of_any_type"] = len(quiet)
    record["built_from"] = {
        "eras": eras,
        "conversion_record_sha256": (
            sha256_file(conversion_record) if conversion_record.exists() else None
        ),
        "incident_types": list(types),
        "configured_first_day": first_day.isoformat() if first_day else None,
        "configured_last_day": last_day.isoformat() if last_day else None,
        "reporting_gaps": [
            {"e_code_territory": code, "start": start.isoformat(), "end": end.isoformat()}
            for code, start, end in gaps
        ],
    }
    record_path = processed / "panel_record.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(
        f"Wrote {record['rows_written']:,} rows to {panel_path}, "
        f"{record['first_day']} to {record['last_day']}, from {record['rows_read']:,} "
        f"incidents read"
    )
    print(f"Wrote the build record to {record_path}")
    print(f"Wrote zero day shares for {len(summary)} series to {shares_path}")
    print(
        f"Wrote {len(quiet)} territory-days with no incidents of any type, outside a "
        f"declared gap, to {quiet_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
