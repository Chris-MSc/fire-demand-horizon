"""Tests for the aggregation of gridded weather to territories.

Grids, label arrays and monthly files are synthetic: a 2 by 3 grid with two
territories and a few days per file. Nothing is read from the archive.
"""

from __future__ import annotations

import calendar
import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from fdh import aggregate
from fdh.grid import X_NAME, Y_NAME

FILL = 1e20
CODES = ["E31000001", "E31000002"]
VARIABLES = ["rainfall", "tasmax", "tasmin"]

# Territory 0 holds three cells, territory 1 holds two, one cell is unassigned.
LABELS = np.array([[0, 0, -1], [1, 1, 0]], dtype=np.int16)
EASTINGS = np.array([100_500.0, 101_500.0, 102_500.0])
NORTHINGS = np.array([20_500.0, 21_500.0])


def write_month(
    directory: Path,
    variable: str,
    month: int,
    values: np.ndarray | None = None,
    units: str | None = None,
) -> Path:
    """Write one monthly file of 2010. `values` defaults to ones on every day and cell."""
    last = calendar.monthrange(2010, month)[1]
    if values is None:
        values = np.ones((last, *LABELS.shape))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{variable}_hadukgrid_uk_1km_day_2010{month:02d}01-2010{month:02d}{last}.nc"
    first = (date(2010, month, 1) - date(1800, 1, 1)).days * 24 + 12
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", values.shape[0])
        dataset.createDimension(Y_NAME, NORTHINGS.size)
        dataset.createDimension(X_NAME, EASTINGS.size)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 1800-01-01 00:00:00"
        time[:] = first + 24 * np.arange(values.shape[0])
        dataset.createVariable(X_NAME, "f8", (X_NAME,))[:] = EASTINGS
        dataset.createVariable(Y_NAME, "f8", (Y_NAME,))[:] = NORTHINGS
        var = dataset.createVariable(variable, "f8", ("time", Y_NAME, X_NAME), fill_value=FILL)
        var.units = units or aggregate.UNITS[variable]
        var.set_auto_mask(False)
        var[:] = values
    return path


def april(day_values: list[list[list[float]]]) -> np.ndarray:
    """Thirty days of ones, with the first days replaced by `day_values`."""
    values = np.ones((30, *LABELS.shape))
    for step, day in enumerate(day_values):
        values[step] = np.array(day, dtype=float)
    return values


def prepared() -> aggregate.Prepared:
    return aggregate.prepare(LABELS, codes=CODES)


def write_archive(directory: Path, overrides: dict[str, np.ndarray] | None = None):
    for variable in VARIABLES:
        write_month(directory, variable, 4, (overrides or {}).get(variable))
    return aggregate.discover(directory, VARIABLES)


def test_the_mean_is_unweighted_over_the_assigned_cells(tmp_path: Path) -> None:
    # The unassigned cell holds 999 and must not reach either mean.
    path = write_month(tmp_path, "rainfall", 4, april([[[1.0, 2.0, 999.0], [10.0, 20.0, 6.0]]]))

    result = aggregate.aggregate_file(path, prepared(), FILL)

    assert result.days[0] == date(2010, 4, 1)
    assert result.means[0].tolist() == [3.0, 15.0]
    assert result.counts[0].tolist() == [3, 2]


def test_fill_and_non_finite_values_are_left_out_of_the_mean_and_the_count(
    tmp_path: Path,
) -> None:
    path = write_month(
        tmp_path,
        "rainfall",
        4,
        april(
            [
                [[1.0, FILL, 0.0], [np.nan, 20.0, 6.0]],
                [[np.inf, 2.0, 0.0], [10.0, -np.inf, 4.0]],
            ]
        ),
    )

    result = aggregate.aggregate_file(path, prepared(), FILL)

    assert result.means[:2].tolist() == [[3.5, 20.0], [3.0, 10.0]]
    assert result.counts[:2].tolist() == [[2, 1], [2, 1]]


def test_a_grid_whose_coordinates_differ_from_the_mask_stops_the_read(tmp_path: Path) -> None:
    path = write_month(tmp_path, "rainfall", 4)
    flipped = aggregate.prepare(
        LABELS, codes=CODES, corners=(100_500.0, 102_500.0, 21_500.0, 20_500.0)
    )

    with pytest.raises(aggregate.AggregateError):
        aggregate.aggregate_file(path, flipped, FILL)


def test_a_territory_with_no_contributing_cells_is_null_not_zero(tmp_path: Path) -> None:
    empty_day = april([[[1.0, 1.0, 1.0], [FILL, FILL, 1.0]]])
    files = write_archive(tmp_path, {name: empty_day for name in VARIABLES})

    rows, record = aggregate.build(files, prepared(), FILL)

    assert len(rows) == 60, "one row per day and territory"
    assert rows[0]["weather_date"] == date(2010, 4, 1), "the published label, not an incident day"
    first_day = {row["e_code_territory"]: row for row in rows[:2]}
    assert first_day[CODES[0]]["rainfall_mm"] == 1.0
    assert first_day[CODES[1]]["rainfall_mm"] is None
    assert first_day[CODES[1]]["tasmax_c"] is None
    assert first_day[CODES[1]]["cells"] == 0
    assert record["null_territory_days"] == 1


def test_variables_that_disagree_on_the_contributing_count_stop_the_build(
    tmp_path: Path,
) -> None:
    files = write_archive(
        tmp_path, {"tasmin": april([[[1.0, 1.0, 1.0], [FILL, 1.0, 1.0]]])}
    )

    with pytest.raises(aggregate.AggregateError):
        aggregate.build(files, prepared(), FILL)


def test_units_other_than_those_the_column_states_stop_the_build(tmp_path: Path) -> None:
    for variable in ("rainfall", "tasmin"):
        write_month(tmp_path, variable, 4)
    write_month(tmp_path, "tasmax", 4, units="K")
    files = aggregate.discover(tmp_path, VARIABLES)

    with pytest.raises(aggregate.AggregateError):
        aggregate.build(files, prepared(), FILL)


def test_days_that_do_not_run_unbroken_stop_the_build(tmp_path: Path) -> None:
    for variable in VARIABLES:
        for month in (4, 5, 6):
            write_month(tmp_path, variable, month)
    files = aggregate.discover(tmp_path, VARIABLES)
    without_may = {name: [items[0], items[2]] for name, items in files.items()}

    with pytest.raises(aggregate.AggregateError):
        aggregate.build(without_may, prepared(), FILL)


def write_record(tmp_path: Path, last_day: str) -> tuple[Path, dict]:
    """Write a build record covering April 2010 to `last_day`, with its basis."""
    labels_path = tmp_path / "grid_labels.npy"
    np.save(labels_path, LABELS)
    expected = aggregate.basis(
        date(2010, 4, 1), date(2010, 5, 31), VARIABLES, "fixture_layer", labels_path
    )
    record_path = tmp_path / aggregate.RECORD_FILE
    record = {**expected, "first_day": "2010-04-01", "last_day": last_day}
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return record_path, expected


def test_a_panel_built_from_another_mask_or_layer_is_stale(tmp_path: Path) -> None:
    record_path, expected = write_record(tmp_path, "2010-05-31")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aggregate.check_current(record_path, expected)
    with pytest.raises(aggregate.AggregateError):
        aggregate.check_current(record_path, {**expected, "boundaries_layer": "another"})
    with pytest.raises(aggregate.AggregateError):
        aggregate.check_current(record_path, {**expected, "grid_labels_sha256": "0" * 64})


def test_a_panel_covering_less_than_the_configured_span_warns_and_carries_on(
    tmp_path: Path,
) -> None:
    record_path, expected = write_record(tmp_path, "2010-04-30")

    with pytest.warns(UserWarning):
        record = aggregate.check_current(record_path, expected)

    assert record["last_day"] == "2010-04-30"
