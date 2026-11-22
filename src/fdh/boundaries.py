"""Fire and rescue authority boundaries and the crosswalk to recorded territories.

Boundaries come from the Office for National Statistics feature service, which
serves them in British National Grid (EPSG:27700), matching the HadUK-Grid
projection. Features are paged through resultOffset.

The crosswalk joins on E code first and on normalised name second, because the
two sources name the same authorities differently and Hampshire and Isle of
Wight carries a different E code in each.

The Rural Urban Classification 2021 of fire and rescue areas comes from the same
service as an attribute table with no geometry, so it is fetched as JSON
attributes rather than GeoJSON. It is keyed on FRA25 codes and covers England
and Wales.

Published field names and value sets live in fdh.contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from urllib.parse import urlencode

import requests

from fdh.contract import (
    CLASSIFICATION_FIELDS,
    CLASSIFICATION_KEY,
    CLASSIFICATION_NAME,
    FRA_CODE_FIELD,
    FRA_NAME_FIELD,
    SETTLEMENT_CLASS,
    SETTLEMENT_CLASSES,
)

PAGE_SIZE = 1000
TIMEOUT = 120

_DROP = re.compile(
    r"\b(fire and rescue (service|authority)|fire authority|fire brigade)\b"
)


class BoundaryError(RuntimeError):
    """Raised when an ONS layer or table does not have the expected structure."""


@dataclass(frozen=True)
class CrosswalkRow:
    e_code_territory: str
    frs_territory: str
    boundary_code: str
    boundary_name: str
    match_basis: str
    name_differs: bool


def normalise(name: str) -> str:
    text = name.lower().replace("&", "and")
    text = _DROP.sub("", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _pages(service_url: str, params: dict[str, str]) -> Iterator[dict]:
    """Yield each page of an ArcGIS feature service query, paged by resultOffset."""
    offset = 0
    while True:
        query = {
            "where": "1=1",
            "outFields": "*",
            **params,
            "resultOffset": str(offset),
            "resultRecordCount": str(PAGE_SIZE),
        }
        response = requests.get(
            f"{service_url.rstrip('/')}/query?{urlencode(query)}", timeout=TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise BoundaryError(f"feature service returned an error: {payload['error']}")

        yield payload
        page = payload.get("features", [])
        if not payload.get("exceededTransferLimit") or not page:
            break
        offset += len(page)


def fetch_layer(service_url: str, out_sr: int = 27700) -> dict:
    """Return a GeoJSON FeatureCollection for an ArcGIS feature service layer."""
    features: list[dict] = []
    crs = None
    params = {"returnGeometry": "true", "outSR": str(out_sr), "f": "geojson"}
    for payload in _pages(service_url, params):
        features.extend(payload.get("features", []))
        crs = crs or payload.get("crs")

    if not features:
        raise BoundaryError(f"{service_url} returned no features")
    return {"type": "FeatureCollection", "crs": crs, "features": features}


def fetch_table(service_url: str) -> list[dict]:
    """Return the attribute rows of an ArcGIS feature service layer, no geometry."""
    rows: list[dict] = []
    for payload in _pages(service_url, {"returnGeometry": "false", "f": "json"}):
        for feature in payload.get("features", []):
            if "attributes" not in feature:
                raise BoundaryError(
                    f"{service_url}: expected each feature to carry 'attributes'. "
                    f"Found keys: {sorted(feature)}"
                )
            rows.append(feature["attributes"])

    if not rows:
        raise BoundaryError(f"{service_url} returned no rows")
    return rows


def save_layer(collection: dict, path: Path) -> dict:
    """Write a FeatureCollection and return its provenance record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collection), encoding="utf-8")
    return {
        "filename": path.name,
        "features": len(collection["features"]),
        "properties": sorted(collection["features"][0].get("properties", {})),
        "crs": collection.get("crs"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def save_table(rows: list[dict], path: Path) -> dict:
    """Write attribute rows as JSON and return their provenance record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return {
        "filename": path.name,
        "features": len(rows),
        "properties": sorted(rows[0]),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def load_classification(path: Path) -> dict[str, dict]:
    """Return the classification rows from a saved table, keyed on FRA25CD.

    Stops if any row lacks an expected field, if a code repeats, or if the
    settlement class holds a value outside the four published classes.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise BoundaryError(f"{path.name}: expected a non-empty list of rows")

    expected = (CLASSIFICATION_KEY, CLASSIFICATION_NAME, *CLASSIFICATION_FIELDS)
    by_code: dict[str, dict] = {}
    unexpected: list[tuple[str, object]] = []

    for number, row in enumerate(rows, start=1):
        missing = [field for field in expected if field not in row]
        if missing:
            raise BoundaryError(
                f"{path.name}, row {number}: missing fields {missing}.\n"
                f"  expected: {list(expected)}\n"
                f"  found:    {sorted(row)}"
            )
        code = row[CLASSIFICATION_KEY]
        if code in by_code:
            raise BoundaryError(f"{path.name}: {CLASSIFICATION_KEY} {code} appears twice")
        if row[SETTLEMENT_CLASS] not in SETTLEMENT_CLASSES:
            unexpected.append((code, row[SETTLEMENT_CLASS]))
        by_code[code] = row

    if unexpected:
        raise BoundaryError(
            f"{path.name}: {SETTLEMENT_CLASS} values outside the published classes.\n"
            f"  expected one of: {sorted(SETTLEMENT_CLASSES)}\n"
            f"  found:           {unexpected}"
        )
    return by_code


def load_layer(path: Path) -> tuple[dict[str, str], str, str]:
    """Return code to name, plus the code and name field names, from a saved layer."""
    collection = json.loads(path.read_text(encoding="utf-8"))
    features = collection.get("features", [])
    if not features:
        raise BoundaryError(f"{path.name}: no features")

    properties = features[0].get("properties", {})
    codes = [field for field in properties if FRA_CODE_FIELD.match(field)]
    names = [field for field in properties if FRA_NAME_FIELD.match(field)]
    if not codes or not names:
        raise BoundaryError(
            f"{path.name}: expected properties matching FRAnnCD and FRAnnNM. "
            f"Found: {sorted(properties)}"
        )
    lookup = {
        feature["properties"][codes[0]]: feature["properties"][names[0]]
        for feature in features
    }
    return lookup, codes[0], names[0]


def territories_from_panel(interim_dir: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Return E code to territory name, and the set of E codes per financial year."""
    import pyarrow.parquet as pq

    files = sorted(interim_dir.glob("*/*.parquet"))
    if not files:
        raise BoundaryError(f"no Parquet files under {interim_dir}")

    names: dict[str, str] = {}
    by_year: dict[str, set[str]] = defaultdict(set)
    for path in files:
        table = pq.read_table(
            path, columns=["FRS_TERRITORY", "E_CODE_TERRITORY", "FINANCIAL_YEAR"]
        )
        for name, code, year in zip(
            table.column("FRS_TERRITORY").to_pylist(),
            table.column("E_CODE_TERRITORY").to_pylist(),
            table.column("FINANCIAL_YEAR").to_pylist(),
            strict=True,
        ):
            names[code] = name
            by_year[year].add(code)
    return names, dict(by_year)


def build_crosswalk(
    territories: dict[str, str], boundaries: dict[str, str]
) -> list[CrosswalkRow]:
    by_name = {normalise(name): code for code, name in boundaries.items()}
    rows: list[CrosswalkRow] = []

    for code, name in sorted(territories.items(), key=lambda item: item[1]):
        if code in boundaries:
            basis, matched = "e_code", code
        elif normalise(name) in by_name:
            basis, matched = "normalised_name", by_name[normalise(name)]
        else:
            basis, matched = "unmatched", ""

        rows.append(
            CrosswalkRow(
                e_code_territory=code,
                frs_territory=name,
                boundary_code=matched,
                boundary_name=boundaries.get(matched, ""),
                match_basis=basis,
                name_differs=bool(matched)
                and normalise(name) != normalise(boundaries[matched]),
            )
        )
    return rows


def unmatched_boundaries(
    rows: list[CrosswalkRow], boundaries: dict[str, str]
) -> list[tuple[str, str]]:
    matched = {row.boundary_code for row in rows if row.boundary_code}
    return [(code, boundaries[code]) for code in sorted(set(boundaries) - matched)]


CROSSWALK_COLUMNS: tuple[str, ...] = (*(field.name for field in fields(CrosswalkRow)), "note")


def write_crosswalk(rows: list[CrosswalkRow], path: Path) -> None:
    """Write the crosswalk, carrying hand-written notes over from the existing file.

    A note on a territory the new crosswalk no longer holds stops the run, so a
    decision taken by hand is never dropped silently.
    """
    notes: dict[str, str] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            notes = {
                record["e_code_territory"]: record["note"]
                for record in csv.DictReader(handle)
                if record.get("note", "").strip()
            }

    orphaned = sorted(set(notes) - {row.e_code_territory for row in rows})
    if orphaned:
        raise BoundaryError(
            f"{path.name}: notes exist for territories the new crosswalk does not hold.\n"
            f"  orphaned: {[(code, notes[code]) for code in orphaned]}\n"
            f"Move or delete these notes by hand before rewriting the file."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CROSSWALK_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**asdict(row), "note": notes.get(row.e_code_territory, "")})