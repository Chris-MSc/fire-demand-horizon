"""Tests for the assignment of grid cells to territories.

Grids and polygons are synthetic: a few 1000 m cells and rectangular
boundaries in British National Grid metres. Nothing is read from the archive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset
from shapely.geometry import box, mapping

from fdh import grid
from fdh.config import BoundaryMerge

FILL = 1e20

CORNWALL = "E31000008"
SCILLY = "E31000039"
DEVON = "E31000011"


def feature(code: str, min_x: float, min_y: float, max_x: float, max_y: float) -> dict:
    return {
        "type": "Feature",
        "properties": {"FRA22CD": code, "FRA22NM": "fixture"},
        "geometry": mapping(box(min_x, min_y, max_x, max_y)),
    }


def centres(first: float, count: int) -> np.ndarray:
    return first + grid.SPACING_METRES * np.arange(count)


def small_grid(land: np.ndarray | None = None) -> tuple[grid.Cells, tuple[int, int]]:
    """A grid of 3 rows by 6 columns with its south-west centre at (100500, 20500)."""
    eastings, northings = centres(100_500.0, 6), centres(20_500.0, 3)
    land = np.ones((3, 6), dtype=bool) if land is None else land
    return grid.land_cells(eastings, northings, land), land.shape


def write_netcdf(path: Path, values: np.ndarray) -> Path:
    eastings, northings = centres(100_500.0, 3), centres(20_500.0, 2)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", values.shape[0])
        dataset.createDimension(grid.Y_NAME, northings.size)
        dataset.createDimension(grid.X_NAME, eastings.size)
        dataset.createVariable(grid.X_NAME, "f8", (grid.X_NAME,))[:] = eastings
        dataset.createVariable(grid.Y_NAME, "f8", (grid.Y_NAME,))[:] = northings
        var = dataset.createVariable(
            "rainfall", "f8", ("time", grid.Y_NAME, grid.X_NAME), fill_value=FILL
        )
        var[:] = values
    return path


def merge(boundary_code: str, e_code_territory: str) -> BoundaryMerge:
    return BoundaryMerge(
        boundary_code=boundary_code,
        boundary_name="fixture",
        e_code_territory=e_code_territory,
        frs_territory="fixture",
        source="fixture",
    )


def test_boundaries_in_degrees_are_refused() -> None:
    with pytest.raises(grid.GridError):
        grid.boundaries_from_features([feature(CORNWALL, -5.7, 49.9, -4.2, 50.9)])


def test_land_is_the_cells_valid_on_every_day(tmp_path: Path) -> None:
    values = np.ones((2, 2, 3))
    values[:, 0, 0] = FILL

    _, _, land = grid.read_grid(write_netcdf(tmp_path / "probe.nc", values), FILL)

    assert land.tolist() == [[False, True, True], [True, True, True]]


def test_validity_that_varies_within_the_file_stops_the_read(tmp_path: Path) -> None:
    values = np.ones((2, 2, 3))
    values[1, 0, 0] = FILL

    with pytest.raises(grid.GridError):
        grid.read_grid(write_netcdf(tmp_path / "probe.nc", values), FILL)


def test_isles_of_scilly_cells_go_to_the_cornwall_code() -> None:
    cells, _ = small_grid()
    boundaries = grid.boundaries_from_features(
        [
            feature(SCILLY, 100_000, 20_000, 102_000, 23_000),
            feature(CORNWALL, 104_000, 20_000, 106_000, 23_000),
        ]
    )

    codes, record = grid.assign(cells, boundaries, {CORNWALL: CORNWALL, SCILLY: CORNWALL})

    assert codes.reshape(3, 6)[0].tolist() == [CORNWALL, CORNWALL, None, None, CORNWALL, CORNWALL]
    assert record["assigned"] == 12
    assert record["outside_every_polygon"] == 6


def test_a_boundary_code_that_differs_from_the_recorded_code_is_mapped() -> None:
    cells, _ = small_grid()
    boundaries = grid.boundaries_from_features(
        [feature("E31000048", 100_000, 20_000, 106_000, 23_000)]
    )

    codes, _ = grid.assign(cells, boundaries, {"E31000048": "E31000017"})

    assert set(codes) == {"E31000017"}


def test_the_label_array_indexes_territories_in_e_code_order() -> None:
    land = np.ones((3, 6), dtype=bool)
    land[2, 5] = False
    cells, grid_shape = small_grid(land)
    boundaries = grid.boundaries_from_features(
        [
            feature(DEVON, 100_000, 20_000, 102_000, 23_000),
            feature(CORNWALL, 104_000, 20_000, 106_000, 23_000),
        ]
    )
    codes, _ = grid.assign(cells, boundaries, {CORNWALL: CORNWALL, DEVON: DEVON})

    labels, index = grid.label_array(grid_shape, cells, codes)

    assert index == [CORNWALL, DEVON]
    assert labels.tolist() == [
        [1, 1, -1, -1, 0, 0],
        [1, 1, -1, -1, 0, 0],
        [1, 1, -1, -1, 0, -1],
    ]


def test_a_mask_built_from_another_crosswalk_or_merge_is_stale(tmp_path: Path) -> None:
    boundaries = tmp_path / "boundaries.geojson"
    crosswalk = tmp_path / "crosswalk.csv"
    boundaries.write_text("polygons", encoding="utf-8")
    crosswalk.write_text("crosswalk", encoding="utf-8")
    merged = [merge(SCILLY, CORNWALL)]
    built = {"built_from": grid.mask_basis("layer", boundaries, crosswalk, merged)}

    grid.check_mask_basis(built, grid.mask_basis("layer", boundaries, crosswalk, merged))
    with pytest.raises(grid.GridError):
        grid.check_mask_basis(built, grid.mask_basis("layer", boundaries, crosswalk, []))
    crosswalk.write_text("crosswalk, one row changed", encoding="utf-8")
    with pytest.raises(grid.GridError):
        grid.check_mask_basis(built, grid.mask_basis("layer", boundaries, crosswalk, merged))


def test_the_crosswalk_is_keyed_on_boundary_code_with_merges_added(tmp_path: Path) -> None:
    path = tmp_path / "crosswalk.csv"
    path.write_text(
        "e_code_territory,frs_territory,boundary_code,boundary_name,match_basis,note\n"
        f"{CORNWALL},fixture,{CORNWALL},fixture,e_code,\n"
        "E31000017,fixture,E31000048,fixture,normalised_name,\n",
        encoding="utf-8",
    )

    lookup = grid.load_crosswalk(path, merged=[merge(SCILLY, CORNWALL)])

    assert lookup == {CORNWALL: CORNWALL, "E31000048": "E31000017", SCILLY: CORNWALL}
