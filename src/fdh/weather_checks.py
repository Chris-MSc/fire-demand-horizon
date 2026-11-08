"""Quality checks on the HadUK-Grid daily files.

Each function returns rows rather than printing, so the results can be tabulated
and written to the quality report. Checks that find a difference return it as a
row; only a structural impossibility raises.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from fdh.weather import ArchiveFile, total_bytes

FILL = 1e20
EPOCH_PREFIX = "hours since "

# A 09:00 observing day written as 0900 or 09:00, not part of a longer number.
STATES_0900 = re.compile(r"(?<!\d)09:?00(?!\d)")

# Plausible daily extremes for England, by variable, in the file's own units.
RANGE_LIMITS: dict[str, tuple[float, float]] = {
    "rainfall": (0.0, 500.0),
    "tas": (-30.0, 45.0),
    "tasmax": (-25.0, 45.0),
    "tasmin": (-35.0, 35.0),
}


# Observing periods as the temperature files state them in cell_methods.
TASMAX_WINDOW = "0900 on day D to 0900 on day D+1"
TASMIN_WINDOW = "0900 on day D-1 to 0900 on day D"


class GridMismatch(RuntimeError):
    """Raised when files do not share a grid, so one mask cannot serve them all."""


class WindowMismatch(RuntimeError):
    """Raised when a file states an observing period other than the one assumed."""


def _measurement(dataset: Dataset):
    """Return the single three-dimensional variable, which holds the measurement."""
    candidates = [var for var in dataset.variables.values() if var.ndim == 3]
    if len(candidates) != 1:
        raise GridMismatch(
            f"expected one three-dimensional variable, found "
            f"{[var.name for var in candidates]}"
        )
    return candidates[0]


def _epoch(units: str) -> datetime:
    if not units.startswith(EPOCH_PREFIX):
        raise GridMismatch(f"unexpected time units: {units}")
    return datetime.fromisoformat(units[len(EPOCH_PREFIX) :].strip())


def describe_file(path: Path) -> dict:
    """Return the metadata contract and grid summary for one file."""
    with Dataset(path) as dataset:
        var = _measurement(dataset)
        time = dataset.variables["time"]
        epoch = _epoch(time.units)
        bounds = dataset.variables["time_bnds"][:]
        starts = [epoch + timedelta(hours=int(row[0])) for row in bounds]
        ends = [epoch + timedelta(hours=int(row[1])) for row in bounds]
        spans = Counter(
            (end - start).total_seconds() / 3600 for start, end in zip(starts, ends, strict=True)
        )
        start_hours = Counter(start.hour for start in starts)

        values = var[:]
        data = np.ma.filled(np.ma.masked_invalid(values), FILL)
        valid = data < FILL / 2

        return {
            "file": path.name,
            "variable": var.name,
            "short_name": dataset.getncattr("short_name"),
            "source": dataset.getncattr("source"),
            "version": dataset.getncattr("version"),
            "units": var.units,
            "cell_methods": getattr(var, "cell_methods", None),
            "comment": getattr(var, "comment", None),
            "fill_value": float(getattr(var, "_FillValue", np.nan)),
            "calendar": getattr(time, "calendar", None),
            "steps": len(starts),
            "first_start": starts[0],
            "last_end": ends[-1],
            "span_hours": sorted(spans),
            "start_hours": sorted(start_hours),
            "shape": tuple(int(n) for n in var.shape),
            "valid_cells": int(valid[0].sum()),
            "valid_share": round(float(valid[0].mean()), 4),
            "valid_cells_vary": bool(len({int(day.sum()) for day in valid}) > 1),
            "min": float(np.nanmin(data[valid])) if valid.any() else None,
            "max": float(np.nanmax(data[valid])) if valid.any() else None,
            "x": dataset.variables["projection_x_coordinate"][:].data,
            "y": dataset.variables["projection_y_coordinate"][:].data,
            "mask": valid[0],
        }


def window_rows(descriptions: list[dict]) -> list[dict]:
    """Compare the window each file's time bounds declare with the one its text states.

    The text is cell_methods and comment together. A file whose text says
    nothing about a 09:00 day is reported as not stated, not as a disagreement.
    """
    rows = []
    for item in descriptions:
        text = " ".join(str(value) for value in (item["cell_methods"], item["comment"]) if value)
        text_0900 = bool(STATES_0900.search(text))
        bounds_0900 = item["start_hours"] == [9]
        if not text_0900:
            agreement = "not stated"
        else:
            agreement = "agree" if bounds_0900 else "disagree"
        rows.append(
            {
                "variable": item["variable"],
                "cell_methods": item["cell_methods"],
                "comment": item["comment"],
                "bounds_start_hours": item["start_hours"],
                "bounds_span_hours": item["span_hours"],
                "text_states_0900": text_0900,
                "bounds_start_0900": bounds_0900,
                "agreement": agreement,
            }
        )
    return rows


def grid_rows(descriptions: list[dict]) -> list[dict]:
    """Check every file shares the first file's grid, so a single mask serves all."""
    reference = descriptions[0]
    return [
        {
            "variable": item["variable"],
            "shape": item["shape"][1:],
            "x_matches": bool(np.array_equal(item["x"], reference["x"])),
            "y_matches": bool(np.array_equal(item["y"], reference["y"])),
            "mask_matches": bool(np.array_equal(item["mask"], reference["mask"])),
            "valid_cells": item["valid_cells"],
            "valid_share": item["valid_share"],
            "mask_varies_within_month": item["valid_cells_vary"],
        }
        for item in descriptions
    ]


def range_rows(descriptions: list[dict]) -> list[dict]:
    """Provenance strings and the plausibility of observed values, by variable."""
    rows = []
    for item in descriptions:
        low, high = RANGE_LIMITS.get(item["variable"], (None, None))
        rows.append(
            {
                "variable": item["variable"],
                "units": item["units"],
                "fill": item["fill_value"],
                "source": item["source"],
                "version": item["version"],
                "calendar": item["calendar"],
                "steps": item["steps"],
                "min": round(item["min"], 2) if item["min"] is not None else None,
                "max": round(item["max"], 2) if item["max"] is not None else None,
                "within_expected": None
                if low is None
                else bool(item["min"] >= low and item["max"] <= high),
                "expected": None if low is None else f"{low} to {high}",
            }
        )
    return rows


def _failing_excess(low: np.ma.MaskedArray, high: np.ma.MaskedArray) -> tuple[int, np.ndarray]:
    """Return the cell-days compared and the amounts by which low exceeds high."""
    valid = ~np.ma.getmaskarray(low) & ~np.ma.getmaskarray(high)
    excess = (np.ma.getdata(low) - np.ma.getdata(high))[valid]
    return int(valid.sum()), excess[excess > 0]


def temperature_ordering(
    tasmin: np.ma.MaskedArray,
    tasmax: np.ma.MaskedArray,
    tasmin_methods: str,
    tasmax_methods: str,
) -> dict:
    """Count cell-days where the minimum exceeds the maximum over the same 24 hours.

    tasmax for day D covers 09:00 D to 09:00 D+1 and tasmin for day D covers
    09:00 D-1 to 09:00 D, so the minimum inside the window of tasmax[D] is
    tasmin[D+1]. The run stops if either file's cell_methods states another
    period. The same-day comparison is returned alongside for reference.
    """
    for name, methods, window in (
        ("tasmin", tasmin_methods, TASMIN_WINDOW),
        ("tasmax", tasmax_methods, TASMAX_WINDOW),
    ):
        if window not in str(methods):
            raise WindowMismatch(
                f"{name} cell_methods does not state the assumed observing period.\n"
                f"  expected to contain: {window!r}\n"
                f"  found:               {methods!r}"
            )
    if tasmin.shape != tasmax.shape:
        raise GridMismatch(f"tasmin shape {tasmin.shape} differs from tasmax {tasmax.shape}")

    compared, failing = _failing_excess(tasmin[1:], tasmax[:-1])
    same_day_compared, same_day_failing = _failing_excess(tasmin, tasmax)
    return {
        "cell_days": compared,
        "failing": int(failing.size),
        "failing_share": failing.size / compared if compared else 0.0,
        "median_excess": float(np.median(failing)) if failing.size else 0.0,
        "max_excess": float(failing.max()) if failing.size else 0.0,
        "same_day_cell_days": same_day_compared,
        "same_day_failing": int(same_day_failing.size),
    }


def contiguity_rows(files_by_variable: dict[str, list[ArchiveFile]]) -> list[dict]:
    """Check the monthly files tile the window without gaps, overlaps or repeats."""
    rows = []
    for variable, items in files_by_variable.items():
        gaps, overlaps = [], []
        for earlier, later in zip(items, items[1:], strict=False):
            expected = earlier.end + timedelta(days=1)
            if later.start > expected:
                gaps.append((earlier.end, later.start))
            elif later.start < expected:
                overlaps.append((earlier.end, later.start))
        months = {(item.start.year, item.start.month) for item in items}
        rows.append(
            {
                "variable": variable,
                "files": len(items),
                "distinct_months": len(months),
                "first": items[0].start,
                "last": items[-1].end,
                "gaps": len(gaps),
                "overlaps": len(overlaps),
                "gap_spans": gaps,
                "overlap_spans": overlaps,
                "gigabytes": round(total_bytes(items) / 1e9, 1),
            }
        )
    return rows


def coverage_rows(files_by_variable: dict[str, list[ArchiveFile]]) -> list[dict]:
    """Check every variable offers the same months, so the panel is balanced.

    Each variable is compared against the months offered by any variable, so a
    month one variable lacks is named against that variable.
    """
    sets = {
        variable: {(item.start.year, item.start.month) for item in items}
        for variable, items in files_by_variable.items()
    }
    offered = set().union(*sets.values())
    return [
        {"variable": variable, "months": len(months), "missing": sorted(offered - months)}
        for variable, months in sets.items()
    ]


def expected_months(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month) + 1
