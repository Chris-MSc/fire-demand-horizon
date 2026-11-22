"""Daily weather per territory from the gridded archive.

The cell mask and the monthly NetCDF files become one row per day per
territory. The rule is the unweighted mean of the land cells assigned to the
territory. Cells are equal area at 1 km, so area weighting gives the same
value; population and incident weighting are the alternatives and are not
used here.

Each day is carried on its published label, in the column weather_date. The
observing windows behind the labels differ between variables
(weather.label_offset_days in config), and no offset is applied here:
alignment with incident days belongs where the covariates are derived. The
column is not named incident_date because the label is not the day the value
is known: rainfall and tasmax labelled D are complete at 09:00 on D+1.

One mask serves every variable, so the number of cells contributing to a
territory on a day must agree across variables. A disagreement stops the
build. A territory with no contributing cells on a day is written as null and
counted.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from fdh.grid import X_NAME, Y_NAME
from fdh.provenance import sha256_file

# A monthly file name ends with the first and last day it covers.
DATED = re.compile(r"^(?P<stem>.+?)(?P<start>\d{8})-(?P<end>\d{8})\.nc$")

# Directory holding single files fetched by the notebooks, outside the archive.
PROBE_DIRECTORY = "probe"

EPOCH_PREFIX = "hours since "

# Panel column and expected units for each variable.
COLUMNS: dict[str, str] = {
    "rainfall": "rainfall_mm",
    "tasmax": "tasmax_c",
    "tasmin": "tasmin_c",
}
UNITS: dict[str, str] = {"rainfall": "mm", "tasmax": "degC", "tasmin": "degC"}

BUILD_COMMAND = "uv run python scripts/build_weather_panel.py"

PANEL_FILE = "weather_panel.parquet"
RECORD_FILE = "weather_panel_record.json"


class AggregateError(RuntimeError):
    """Raised when the archive or the mask cannot produce the weather panel."""


@dataclass(frozen=True)
class MonthFile:
    """One monthly file: the variable and the days its name says it covers."""

    variable: str
    start: date
    end: date
    path: Path

    @property
    def month(self) -> str:
        return f"{self.start.year}-{self.start.month:02d}"


@dataclass(frozen=True)
class Prepared:
    """Flat positions of the assigned cells, their labels, and the grid they fit."""

    positions: np.ndarray
    labels: np.ndarray
    assigned: np.ndarray
    shape: tuple[int, int]
    codes: tuple[str, ...]
    corners: tuple[float, float, float, float] | None = None

    @property
    def territories(self) -> int:
        return len(self.codes)


@dataclass(frozen=True)
class FileResult:
    """Per-day, per-territory means and contributing cell counts for one file."""

    units: str | None
    days: list[date]
    means: np.ndarray
    counts: np.ndarray


def _months_between(first: str, last: str) -> list[str]:
    year, month = (int(part) for part in first.split("-"))
    months = []
    while f"{year}-{month:02d}" <= last:
        months.append(f"{year}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def discover(raw_weather_dir: Path, variables: list[str]) -> dict[str, list[MonthFile]]:
    """Return the monthly files for each variable, in date order.

    The variable is the leading part of the file name and the month comes from
    the two dates that close it. Files of other variables are left out, as is
    anything under a probe directory.

    Stops unless every variable holds each month of one unbroken span exactly
    once. `build` makes the same demand of the days it reads, at the end of
    the run; this comparison of the names costs nothing and fails before the
    archive is read.
    """
    found: dict[str, list[MonthFile]] = {name: [] for name in variables}
    by_length = sorted(variables, key=len, reverse=True)
    for path in sorted(raw_weather_dir.rglob("*.nc")):
        if PROBE_DIRECTORY in path.relative_to(raw_weather_dir).parts[:-1]:
            continue
        variable = next((name for name in by_length if path.name.startswith(f"{name}_")), None)
        if variable is None:
            continue
        match = DATED.match(path.name)
        start = datetime.strptime(match.group("start"), "%Y%m%d").date()
        end = datetime.strptime(match.group("end"), "%Y%m%d").date()
        found[variable].append(MonthFile(variable, start, end, path))

    held = {}
    for name, items in found.items():
        items.sort(key=lambda item: item.start)
        held[name] = [item.month for item in items]
    every = sorted({month for months in held.values() for month in months})
    whole = _months_between(every[0], every[-1]) if every else []
    uneven = {
        name: f"{len(months)} files, absent {sorted(set(whole) - set(months))}, "
        f"held twice {sorted({month for month in months if months.count(month) > 1})}"
        for name, months in held.items()
        if months != whole
    }
    if uneven or not every:
        raise AggregateError(
            f"the monthly files under {raw_weather_dir} do not tile one span for "
            f"every variable.\n"
            f"  expected: each of {len(whole)} months once for each of {variables}\n"
            f"  found:    {uneven or 'no files'}"
        )
    return found


def prepare(
    labels: np.ndarray,
    codes: list[str] | None = None,
    corners: tuple[float, float, float, float] | None = None,
) -> Prepared:
    """Return the flat positions and labels of the assigned cells.

    Each day then reads the assigned cells only rather than the whole grid.
    `codes` gives the E code for each label, in label order, and defaults to
    the label numbers. `corners` holds the first and last easting and the
    first and last northing of the grid the labels were built on; when given,
    every file is checked against them.
    """
    flat = labels.ravel()
    positions = np.flatnonzero(flat >= 0)
    if positions.size == 0:
        raise AggregateError("label array assigns no cells to any territory")
    cell_labels = flat[positions].astype(np.int64)
    territories = int(cell_labels.max()) + 1
    assigned = np.bincount(cell_labels, minlength=territories)

    names = tuple(codes) if codes is not None else tuple(str(n) for n in range(territories))
    if len(names) != territories or (assigned == 0).any():
        raise AggregateError(
            "labels and territory codes do not line up.\n"
            f"  expected: {len(names)} codes, each with at least one cell\n"
            f"  found:    labels 0 to {territories - 1}, "
            f"{int((assigned == 0).sum())} with no cells"
        )
    return Prepared(
        positions=positions,
        labels=cell_labels,
        assigned=assigned,
        shape=(int(labels.shape[0]), int(labels.shape[1])),
        codes=names,
        corners=corners,
    )


def _days(time) -> list[date]:
    """Return the calendar day of each step of a NetCDF time variable."""
    units = str(time.units)
    if not units.startswith(EPOCH_PREFIX):
        raise AggregateError(f"unexpected time units: {units}")
    epoch = datetime.fromisoformat(units[len(EPOCH_PREFIX) :].strip())
    return [(epoch + timedelta(hours=float(hours))).date() for hours in time[:]]


def aggregate_file(path: Path, prepared: Prepared, fill_value: float) -> FileResult:
    """Return the mean and the contributing cell count per territory for each day.

    A cell is left out of a day when its value equals the fill value or is not
    finite. The mean is NaN where no cell contributes.

    The file is read whole, once. The archive stores each variable in
    compressed chunks ten days deep, and one day of the grid spans more chunk
    bytes than the chunk cache holds, so a read of one day at a time
    decompresses the same chunks for every day in them. The assigned cells are
    taken before the values are filled or cast, so only those cells are
    copied.
    """
    with Dataset(path) as dataset:
        measurements = [var for var in dataset.variables.values() if var.ndim == 3]
        if len(measurements) != 1:
            raise AggregateError(
                f"{path.name}: expected one three-dimensional variable, found "
                f"{[var.name for var in measurements]}"
            )
        var = measurements[0]
        if tuple(var.shape[1:]) != prepared.shape:
            raise AggregateError(
                f"{path.name}: grid does not match the label array.\n"
                f"  expected: {prepared.shape}\n"
                f"  found:    {tuple(var.shape[1:])}"
            )
        if prepared.corners is not None:
            eastings = dataset.variables[X_NAME]
            northings = dataset.variables[Y_NAME]
            corners = (
                float(eastings[0]),
                float(eastings[-1]),
                float(northings[0]),
                float(northings[-1]),
            )
            if corners != tuple(float(value) for value in prepared.corners):
                raise AggregateError(
                    f"{path.name}: grid coordinates do not match the label array.\n"
                    f"  expected first and last easting, first and last northing: "
                    f"{prepared.corners}\n"
                    f"  found:    {corners}"
                )

        days = _days(dataset.variables["time"])
        territories = prepared.territories
        means = np.full((len(days), territories), np.nan)
        counts = np.zeros((len(days), territories), dtype=np.int32)
        whole = var[:]
        assigned = whole.reshape(whole.shape[0], -1)[:, prepared.positions]
        del whole
        cells = np.asarray(np.ma.filled(assigned, fill_value), dtype=float)
        for step in range(len(days)):
            values = cells[step]
            valid = np.isfinite(values) & ~np.isclose(values, fill_value)
            held = np.bincount(prepared.labels[valid], minlength=territories)
            sums = np.bincount(
                prepared.labels[valid], weights=values[valid], minlength=territories
            )
            counts[step] = held
            np.divide(sums, held, out=means[step], where=held > 0)

        return FileResult(
            units=getattr(var, "units", None), days=days, means=means, counts=counts
        )


def build(
    files: dict[str, list[MonthFile]],
    prepared: Prepared,
    fill_value: float,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[dict], dict]:
    """Return one row per day and territory, with a record of what was built.

    Files are read one month at a time and only the per-day, per-territory
    results are kept. Each file must carry the units its panel column states.
    The contributing cell count must agree across variables for every day and
    territory; a disagreement stops the build naming the first day and
    territory. A territory with no contributing cells on a day has null in
    every variable for that day.

    The files read together must hold the same days, and the rows must hold
    every day from the first file's start to the last file's end exactly once
    and in order. Those two conditions cover a missing month, a month read
    twice and files out of step between variables, any of which would
    otherwise leave a panel that joins to the incident days with gaps or
    repeats.
    """
    variables = list(files)
    lengths = {name: len(items) for name, items in files.items()}

    rows: list[dict] = []
    days_written: list[date] = []
    null_examples: list[dict] = []
    below_examples: list[dict] = []
    nulls = below = 0
    days_below: set[date] = set()
    total = lengths[variables[0]]

    for number in range(total):
        month_files = [files[name][number] for name in variables]
        if progress:
            progress(month_files[0].month, number + 1, total)

        results = {}
        for item in month_files:
            results[item.variable] = aggregate_file(item.path, prepared, fill_value)
            if results[item.variable].units != UNITS[item.variable]:
                raise AggregateError(
                    f"{item.path.name}: units differ from those the panel column states.\n"
                    f"  expected: {UNITS[item.variable]}\n"
                    f"  found:    {results[item.variable].units}"
                )

        reference = results[variables[0]]
        days_written.extend(reference.days)
        for name in variables[1:]:
            if results[name].days != reference.days:
                raise AggregateError(
                    "files read together do not hold the same days.\n"
                    f"  expected: {month_files[0].path.name} and "
                    f"{files[name][number].path.name} to cover one month\n"
                    f"  found:    {reference.days[0]} to {reference.days[-1]} and "
                    f"{results[name].days[0]} to {results[name].days[-1]}"
                )
            differs = np.argwhere(results[name].counts != reference.counts)
            if differs.size:
                step, label = (int(value) for value in differs[0])
                found = {key: int(results[key].counts[step, label]) for key in variables}
                raise AggregateError(
                    "contributing cell counts differ between variables, so one mask "
                    "does not serve them all.\n"
                    f"  expected: the same count for each of {variables}\n"
                    f"  found:    {reference.days[step]}, territory "
                    f"{prepared.codes[label]}: {found}\n"
                    f"  territory-days disagreeing in {month_files[0].month} between "
                    f"{variables[0]} and {name}: {len(differs)}"
                )

        for step, day in enumerate(reference.days):
            for label, code in enumerate(prepared.codes):
                cells = int(reference.counts[step, label])
                row: dict = {"weather_date": day, "e_code_territory": code}
                for name in variables:
                    row[COLUMNS[name]] = (
                        float(results[name].means[step, label]) if cells else None
                    )
                row["cells"] = cells
                rows.append(row)

                if cells < int(prepared.assigned[label]):
                    below += 1
                    days_below.add(day)
                    example = {
                        "weather_date": day.isoformat(),
                        "e_code_territory": code,
                        "cells": cells,
                        "assigned": int(prepared.assigned[label]),
                    }
                    if len(below_examples) < 5:
                        below_examples.append(example)
                    if not cells:
                        nulls += 1
                        if len(null_examples) < 5:
                            null_examples.append(example)

    first = files[variables[0]][0]
    last = files[variables[0]][-1]
    whole_span = [
        first.start + timedelta(days=offset) for offset in range((last.end - first.start).days + 1)
    ]
    if days_written != whole_span:
        absent = sorted(set(whole_span) - set(days_written))
        raise AggregateError(
            "the panel does not hold each day of its span once, in order.\n"
            f"  expected: {len(whole_span):,} days, {first.start} to {last.end}\n"
            f"  found:    {len(days_written):,} days written, "
            f"{len(set(days_written)):,} distinct, {len(absent):,} absent"
            + (f", the first absent {absent[0]}" if absent else "")
        )
    record = {
        "first_day": first.start.isoformat(),
        "last_day": last.end.isoformat(),
        "days": len(rows) // prepared.territories,
        "rows": len(rows),
        "territories": prepared.territories,
        "variables": variables,
        "columns": [COLUMNS[name] for name in variables],
        "files_per_variable": lengths,
        "assigned_cells": int(prepared.assigned.sum()),
        "territory_days_below_assigned": below,
        "days_below_assigned": len(days_below),
        "below_assigned_examples": below_examples,
        "null_territory_days": nulls,
        "null_examples": null_examples,
    }
    return rows, record


def basis(
    window_start: date,
    release_covers_to: date,
    variables: list[str],
    boundaries_layer: str,
    labels_path: Path,
) -> dict:
    """Return what a panel built now would be built from, as written to the record."""
    return {
        "configured_first_day": window_start.isoformat(),
        "configured_last_day": release_covers_to.isoformat(),
        "variables": list(variables),
        "boundaries_layer": boundaries_layer,
        "grid_labels_sha256": sha256_file(labels_path),
    }


def check_current(record_path: Path, expected: dict) -> dict:
    """Return the build record, stopping if the panel was built from other inputs.

    `expected` is the result of `basis` for the current configuration. A panel
    built from another cell mask, other variables, another boundaries layer or
    another configured span holds values the configuration no longer asks
    for, and nothing in the panel itself would show it. A panel that covers
    less than the configured span is incomplete rather than wrong, so it warns
    and carries on.
    """
    record = json.loads(record_path.read_text(encoding="utf-8"))
    differs = [key for key in expected if record.get(key) != expected[key]]
    if differs:
        lines = ["the weather panel was built from other inputs than the configuration names now."]
        for key in differs:
            lines.append(f"  {key}: panel built from {record.get(key)}, now {expected[key]}")
        lines.append(f"Rebuild it from the repository root with: {BUILD_COMMAND}")
        raise AggregateError("\n".join(lines))

    covered = (record["first_day"], record["last_day"])
    configured = (expected["configured_first_day"], expected["configured_last_day"])
    if covered != configured:
        warnings.warn(
            f"the weather panel covers {covered[0]} to {covered[1]} of the configured "
            f"{configured[0]} to {configured[1]}. Rebuilding takes in any further "
            f"files: {BUILD_COMMAND}",
            stacklevel=2,
        )
    return record
