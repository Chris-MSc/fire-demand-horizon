"""Build the daily weather panel from the downloaded HadUK-Grid archive.

Writes one row per day and territory across the whole downloaded span: the
unweighted mean of the land cells assigned to the territory, for each
configured variable, with the number of contributing cells. The span is not
cut to the incident panel, since the months before it exist for accumulated
covariates to reach back into and truncation belongs where those are derived.

Reads grid_labels.npy and its index, written by notebook 01, and stops if the
mask was built from another boundary layer, crosswalk or set of merges than
the configuration names now. Writes weather_panel.parquet and
weather_panel_record.json to the processed directory. The record holds what
the panel was built from, which notebook 04 compares against the current
configuration, and weather.label_offset_days, since tasmax_c and tasmin_c on
one row do not describe the same 24 hours.

Usage:
    uv run python scripts/build_weather_panel.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fdh import aggregate, grid
from fdh.aggregate import PANEL_FILE, RECORD_FILE
from fdh.config import boundary_layer, load_config, repo_root, require
from fdh.grid import LABELS_FILE, LABELS_INDEX_FILE

PANEL_SCHEMA = pa.schema(
    [
        ("weather_date", pa.date32()),
        ("e_code_territory", pa.string()),
        ("rainfall_mm", pa.float32()),
        ("tasmax_c", pa.float32()),
        ("tasmin_c", pa.float32()),
        ("cells", pa.int32()),
    ]
)


def write_panel(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [pa.array([row[field.name] for row in rows], type=field.type) for field in PANEL_SCHEMA],
        schema=PANEL_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")


def report_progress(month: str, number: int, total: int) -> None:
    if number == 1 or number == total or month.endswith("-01"):
        print(f"  {month}  month {number} of {total}", flush=True)


def main() -> int:
    config = load_config()
    raw_dir = config.paths.resolved("raw_dir")
    processed = config.paths.resolved("processed_dir")
    variables = require(config.weather.variables, "weather.variables")
    fill_value = require(config.weather.fill_value, "weather.fill_value")

    labels_path = processed / LABELS_FILE
    labels = np.load(labels_path)
    index = json.loads((processed / LABELS_INDEX_FILE).read_text(encoding="utf-8"))
    layer = boundary_layer(config)
    grid.check_mask_basis(
        index,
        grid.mask_basis(
            layer.era,
            raw_dir / "boundaries" / layer.filename,
            repo_root()
            / require(config.incident_types.crosswalk_file, "incident_types.crosswalk_file"),
            config.boundaries.merged,
        ),
    )
    built_from = aggregate.basis(
        config.weather.window_start,
        config.weather.release_covers_to,
        variables,
        config.boundaries.layer,
        labels_path,
    )
    prepared = aggregate.prepare(
        labels,
        codes=index["index_to_e_code_territory"],
        corners=(
            index["first_easting"],
            index["last_easting"],
            index["first_northing"],
            index["last_northing"],
        ),
    )

    files = aggregate.discover(raw_dir / "weather", variables)
    print(f"Aggregating {sum(len(items) for items in files.values())} monthly files")
    rows, record = aggregate.build(files, prepared, fill_value, progress=report_progress)

    panel_path = processed / PANEL_FILE
    write_panel(rows, panel_path)

    record.update(built_from)
    record["mask_built_from"] = index["built_from"]
    record["label_offset_days"] = config.weather.label_offset_days
    record["built_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["panel_file"] = panel_path.name
    (processed / RECORD_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"\nWrote {record['rows']:,} rows to {panel_path}")
    print(
        f"  span: {record['first_day']} to {record['last_day']}, configured "
        f"{record['configured_first_day']} to {record['configured_last_day']}"
    )
    if record["territory_days_below_assigned"]:
        print(
            f"  territory-days below the assigned cell count: "
            f"{record['territory_days_below_assigned']:,}, of which written as null: "
            f"{record['null_territory_days']:,}"
        )
    print(f"Wrote the build record to {processed / RECORD_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
