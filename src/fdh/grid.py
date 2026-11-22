"""Assignment of HadUK-Grid cells to fire and rescue service areas.

The grid geometry is identical across variables and months, so the assignment
is built once from a reference file and applies to the whole archive. A land
cell belongs to the territory whose boundary polygon contains its centre.

Boundary polygons carry boundary codes, which differ from the E codes recorded
in the incident data for some territories. The matched boundary code is mapped
through the service crosswalk, so the result carries E_CODE_TERRITORY.

Three conditions stop the build: boundary coordinates outside the British
National Grid metre range, a grid spacing other than 1000 m, and a territory
that receives no cells. Land cells outside every polygon and cells matched to
a boundary code with no crosswalk entry are counted and returned, since the
grid covers the whole United Kingdom and the boundary layer covers England
only.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from netCDF4 import Dataset
from shapely.geometry import shape

from fdh.config import BoundaryMerge
from fdh.contract import FRA_CODE_FIELD
from fdh.provenance import sha256_file

CRS = "EPSG:27700"
SPACING_METRES = 1000.0

# Extent of the British National Grid, in metres.
EASTING_RANGE: tuple[float, float] = (0.0, 700_000.0)
NORTHING_RANGE: tuple[float, float] = (0.0, 1_300_000.0)

X_NAME = "projection_x_coordinate"
Y_NAME = "projection_y_coordinate"

UNASSIGNED = -1

LABELS_FILE = "grid_labels.npy"
LABELS_INDEX_FILE = "grid_labels_index.json"


class GridError(RuntimeError):
    """Raised when the grid or the boundaries cannot support the assignment."""


@dataclass(frozen=True)
class Cells:
    """Land cells: position in the grid and centre coordinates in metres."""

    rows: np.ndarray
    cols: np.ndarray
    eastings: np.ndarray
    northings: np.ndarray

    def __len__(self) -> int:
        return int(self.rows.size)


def check_spacing(eastings: np.ndarray, northings: np.ndarray) -> None:
    """Stop unless cell centres are 1000 m apart in both directions."""
    for name, values in (("eastings", eastings), ("northings", northings)):
        steps = np.unique(np.diff(np.asarray(values, dtype=float)))
        if steps.size != 1 or abs(steps[0]) != SPACING_METRES:
            raise GridError(
                f"grid spacing of the {name} is not {SPACING_METRES:.0f} m throughout.\n"
                f"  expected: every step {SPACING_METRES:.0f}\n"
                f"  found:    {steps[:10].tolist()}"
            )


def read_grid(path: Path, fill_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cell centre eastings, cell centre northings and the land mask.

    Land is where the value is not the fill value. The mask holds the cells
    valid on every day in the file, and the read stops if any cell is valid on
    some days and not on others.
    """
    with Dataset(path) as dataset:
        measurements = [var for var in dataset.variables.values() if var.ndim == 3]
        if len(measurements) != 1:
            raise GridError(
                f"{path.name}: expected one three-dimensional variable, found "
                f"{[var.name for var in measurements]}"
            )
        var = measurements[0]
        eastings = np.asarray(dataset.variables[X_NAME][:], dtype=float)
        northings = np.asarray(dataset.variables[Y_NAME][:], dtype=float)
        if var.shape[1:] != (northings.size, eastings.size):
            raise GridError(
                f"{path.name}: measurement shape does not match the coordinates.\n"
                f"  expected: (days, {northings.size}, {eastings.size})\n"
                f"  found:    {tuple(var.shape)}"
            )

        every_day = np.ones(var.shape[1:], dtype=bool)
        any_day = np.zeros(var.shape[1:], dtype=bool)
        for day in range(var.shape[0]):
            values = np.ma.filled(var[day], fill_value)
            valid = np.isfinite(values) & ~np.isclose(values, fill_value)
            every_day &= valid
            any_day |= valid

    varying = int((any_day & ~every_day).sum())
    if varying:
        raise GridError(
            f"{path.name}: validity varies within the file, so it gives no single "
            f"land mask.\n"
            f"  expected: every cell valid on all days or on none\n"
            f"  found:    {varying:,} cells valid on some days only"
        )

    check_spacing(eastings, northings)
    return eastings, northings, every_day


def land_cells(eastings: np.ndarray, northings: np.ndarray, land: np.ndarray) -> Cells:
    """Return row index, column index, easting and northing of the land cells."""
    rows, cols = np.nonzero(land)
    return Cells(
        rows=rows,
        cols=cols,
        eastings=np.asarray(eastings, dtype=float)[cols],
        northings=np.asarray(northings, dtype=float)[rows],
    )


def check_metre_range(bounds: tuple[float, float, float, float]) -> None:
    """Stop unless the bounds sit in the British National Grid range, in metres.

    Coordinates in degrees for England fall inside the same numeric range on
    the northing axis, so bounds no larger than a longitude and latitude are
    refused as well.
    """
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    in_range = (
        EASTING_RANGE[0] <= min_x
        and max_x <= EASTING_RANGE[1]
        and NORTHING_RANGE[0] <= min_y
        and max_y <= NORTHING_RANGE[1]
    )
    degrees = max(abs(min_x), abs(max_x)) <= 180 and max(abs(min_y), abs(max_y)) <= 90
    if not in_range or degrees:
        raise GridError(
            "boundary coordinates are not British National Grid metres.\n"
            f"  expected: eastings within {EASTING_RANGE}, northings within "
            f"{NORTHING_RANGE}, and larger than degrees\n"
            f"  found:    eastings {min_x} to {max_x}, northings {min_y} to {max_y}"
        )


def boundaries_from_features(features: list[dict]) -> gpd.GeoDataFrame:
    """Return boundary code and polygon per feature, in EPSG:27700.

    The coordinate reference system is set rather than read, then the
    coordinates are checked against the metre range.
    """
    if not features:
        raise GridError("boundary layer holds no features")
    properties = features[0].get("properties", {})
    code_fields = [name for name in properties if FRA_CODE_FIELD.match(name)]
    if len(code_fields) != 1:
        raise GridError(
            "expected one boundary property matching FRAnnCD.\n"
            f"  found: {sorted(properties)}"
        )

    frame = gpd.GeoDataFrame(
        {"boundary_code": [item["properties"][code_fields[0]] for item in features]},
        geometry=[shape(item["geometry"]) for item in features],
        crs=CRS,
    )
    repeated = sorted(frame.boundary_code[frame.boundary_code.duplicated()].unique())
    if repeated:
        raise GridError(f"boundary codes appear on more than one feature: {repeated}")
    check_metre_range(tuple(frame.total_bounds))
    return frame


def load_boundaries(path: Path) -> gpd.GeoDataFrame:
    """Return the boundary polygons from a saved GeoJSON layer."""
    collection = json.loads(path.read_text(encoding="utf-8"))
    return boundaries_from_features(collection.get("features", []))


def load_crosswalk(
    path: Path, merged: list[BoundaryMerge] | None = None
) -> dict[str, str]:
    """Return boundary code to E_CODE_TERRITORY from the service crosswalk.

    `merged` holds the merges declared under boundaries.merged in config: each
    adds a boundary polygon that has no recorded territory of its own and
    gives the territory it is folded into. A merge whose territory has no
    crosswalk row stops the load, since the cells would be assigned to a
    territory that is not recorded. A territory with no matched boundary stops
    the load as well.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    unmatched = [row["e_code_territory"] for row in rows if not row["boundary_code"]]
    if unmatched:
        raise GridError(
            f"{path.name}: territories with no boundary code cannot receive cells.\n"
            f"  found: {unmatched}"
        )

    lookup: dict[str, str] = {}
    for row in rows:
        if row["boundary_code"] in lookup:
            raise GridError(
                f"{path.name}: boundary code {row['boundary_code']} appears twice"
            )
        lookup[row["boundary_code"]] = row["e_code_territory"]

    territories = set(lookup.values())
    for merge in merged or []:
        if merge.boundary_code in lookup:
            raise GridError(
                f"merged boundary {merge.boundary_code} ({merge.boundary_name}) "
                f"already has a crosswalk row, for {lookup[merge.boundary_code]}"
            )
        if merge.e_code_territory not in territories:
            raise GridError(
                f"merged boundary {merge.boundary_code} ({merge.boundary_name}) "
                f"names a territory with no crosswalk row, so it is not a "
                f"recorded territory.\n"
                f"  expected one of: {sorted(territories)}\n"
                f"  found:           {merge.e_code_territory} ({merge.frs_territory})"
            )
        lookup[merge.boundary_code] = merge.e_code_territory
    return lookup


def assign(
    cells: Cells, boundaries: gpd.GeoDataFrame, crosswalk: dict[str, str]
) -> tuple[np.ndarray, dict]:
    """Return the E_CODE_TERRITORY of each cell, with a record of what was left out.

    A cell is matched to the polygon containing its centre and the boundary
    code is mapped through `crosswalk`, boundary code to E_CODE_TERRITORY.
    `boundaries` comes from `boundaries_from_features`, which has already
    checked that its coordinates are British National Grid metres.

    Cells outside every polygon and cells whose boundary code has no crosswalk
    entry are left as None and counted in the record. A cell centre inside two
    polygons, or a crosswalk territory that receives no cells, stops the build.
    """
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(cells.eastings, cells.northings), crs=CRS
    )
    joined = gpd.sjoin(
        points,
        boundaries[["boundary_code", "geometry"]].set_crs(CRS, allow_override=True),
        how="left",
        predicate="within",
    )

    twice = int(joined.index.duplicated().sum())
    if twice:
        raise GridError(
            "cell centres fall inside more than one boundary polygon.\n"
            "  expected: 0\n"
            f"  found:    {twice:,}"
        )

    matched = joined["boundary_code"].to_numpy(dtype=object)
    inside = np.array([isinstance(code, str) for code in matched], dtype=bool)
    codes = np.full(len(cells), None, dtype=object)
    no_entry: dict[str, int] = {}
    for boundary_code in sorted(set(matched[inside])):
        selected = inside & (matched == boundary_code)
        if boundary_code in crosswalk:
            codes[selected] = crosswalk[boundary_code]
        else:
            no_entry[boundary_code] = int(selected.sum())

    assigned = {code for code in codes if code is not None}
    empty = sorted(set(crosswalk.values()) - assigned)
    if empty:
        raise GridError(
            "territories in the crosswalk received no cells, so the assignment "
            "is wrong.\n"
            f"  expected: at least one cell for each of {len(set(crosswalk.values()))} "
            f"territories\n"
            f"  found none for: {empty}"
        )

    record = {
        "land_cells": len(cells),
        "assigned": int(sum(code is not None for code in codes)),
        "outside_every_polygon": int((~inside).sum()),
        "no_crosswalk_entry": no_entry,
        "territories": len(assigned),
    }
    return codes, record


def label_array(
    grid_shape: tuple[int, int], cells: Cells, codes: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Return a service index per grid cell and the index to E code mapping.

    The array is int16 with -1 where no service is assigned. Position in the
    returned list is the index, in E code order.
    """
    index = sorted({code for code in codes if code is not None})
    position = {code: number for number, code in enumerate(index)}
    labels = np.full(grid_shape, UNASSIGNED, dtype=np.int16)
    for row, col, code in zip(cells.rows, cells.cols, codes, strict=True):
        if code is not None:
            labels[row, col] = position[code]
    return labels, index


def summarise(cells: Cells, codes: np.ndarray) -> list[dict]:
    """Return the cell count and extent of each territory, in E code order."""
    rows = []
    for code in sorted({code for code in codes if code is not None}):
        selected = codes == code
        rows.append(
            {
                "e_code_territory": code,
                "cells": int(selected.sum()),
                "min_easting": float(cells.eastings[selected].min()),
                "max_easting": float(cells.eastings[selected].max()),
                "min_northing": float(cells.northings[selected].min()),
                "max_northing": float(cells.northings[selected].max()),
            }
        )
    return rows


def mask_basis(
    layer: str, boundaries_path: Path, crosswalk_path: Path, merged: list[BoundaryMerge]
) -> dict:
    """Return what a cell mask built now would be built from.

    The boundary layer by name and by the hash of its saved file, the hash of
    the service crosswalk, and the merges declared in config. The mask depends
    on all four and on nothing else but the grid.
    """
    return {
        "boundaries_layer": layer,
        "boundaries_sha256": sha256_file(boundaries_path),
        "crosswalk_sha256": sha256_file(crosswalk_path),
        "merged_boundaries": {item.boundary_code: item.e_code_territory for item in merged},
    }


def check_mask_basis(index: dict, expected: dict) -> None:
    """Stop unless the saved mask was built from the inputs the configuration names now.

    `index` is the saved label index and `expected` the result of `mask_basis`
    for the current configuration and files. A mask built from another
    boundary layer, crosswalk or set of merges assigns cells differently while
    keeping the same shape, so nothing downstream would show the difference.
    """
    built = index.get("built_from")
    if built is None:
        raise GridError(
            f"{LABELS_INDEX_FILE} does not record what the cell mask was built from.\n"
            "  expected: a built_from entry, written by notebook 01\n"
            "  found:    none. Run notebook 01 again to rebuild the mask."
        )
    differs = [key for key in expected if built.get(key) != expected[key]]
    if differs:
        lines = [
            "the cell mask was built from other inputs than the configuration names now."
        ]
        for key in differs:
            lines.append(f"  {key}: mask built from {built.get(key)}, now {expected[key]}")
        lines.append("Run notebook 01 again to rebuild the mask.")
        raise GridError("\n".join(lines))


def save_labels(
    labels: np.ndarray,
    index: list[str],
    eastings: np.ndarray,
    northings: np.ndarray,
    reference: Path,
    directory: Path,
    built_from: dict,
) -> tuple[Path, Path]:
    """Write the label array and its index mapping, returning both paths.

    The mapping file carries the grid shape and corner cell centres with the
    name of the reference file, so the array can be checked against a weather
    file before it is applied. `built_from` is the result of `mask_basis` and
    is written alongside, so a mask built from other boundaries or another
    crosswalk can be told from the current one.
    """
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / LABELS_FILE
    index_path = directory / LABELS_INDEX_FILE
    np.save(array_path, labels)
    index_path.write_text(
        json.dumps(
            {
                "array_file": array_path.name,
                "dtype": str(labels.dtype),
                "shape": list(labels.shape),
                "unassigned": UNASSIGNED,
                "index_to_e_code_territory": index,
                "first_easting": float(eastings[0]),
                "last_easting": float(eastings[-1]),
                "first_northing": float(northings[0]),
                "last_northing": float(northings[-1]),
                "spacing_metres": SPACING_METRES,
                "crs": CRS,
                "reference_file": reference.name,
                "built_from": built_from,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return array_path, index_path


def write_summary(rows: list[dict], path: Path) -> None:
    """Write the per-territory cell counts as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
