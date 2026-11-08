"""Tests that the ingest scripts read the incident collection only.

config.yaml holds more than one collection under `sources`. The boundary layers
are GeoJSON, so convert_raw.py and build_manifest.py must never open them. The
fixture config has two collections, and the boundary file exists on disk but is
not an ODS file: if either script touched it, the ODS reader would raise.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pyarrow.parquet as pq
import pytest
import yaml
from test_ods import cell, make_ods, row

from fdh.config import INCIDENT_COLLECTION, Config, incident_sources, repo_root
from fdh.contract import EXPECTED_COLUMNS

INCIDENT_FILE = "incidents.ods"
BOUNDARY_FILE = "boundaries.geojson"
DATA_ROW = (
    "Fixture", "E00000000", "2023-24", "2023", "April", "4", "1", "Sat", "Dwellings"
)
REPEATS = 3


def load_script(name: str) -> ModuleType:
    path = repo_root() / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def two_collection_config(tmp_path: Path) -> Config:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    header = row("".join(cell(name) for name in EXPECTED_COLUMNS))
    data = row("".join(cell(value) for value in DATA_ROW), repeat=REPEATS)
    make_ods(
        raw_dir / INCIDENT_FILE,
        '<table:table table:name="202324">' + header + data + "</table:table>"
        '<table:table table:name="config">'
        + row(cell("Pubdate") + cell("fixture"))
        + "</table:table>",
    )
    (raw_dir / BOUNDARY_FILE).write_text('{"type": "FeatureCollection"}', encoding="utf-8")

    settings = yaml.safe_load(
        (repo_root() / "config" / "config.yaml").read_text(encoding="utf-8")
    )
    settings["paths"] = {
        name: str(tmp_path / folder)
        for name, folder in [
            ("raw_dir", "raw"),
            ("interim_dir", "interim"),
            ("processed_dir", "processed"),
            ("reports_dir", "reports"),
        ]
    }
    settings["sources"] = {
        INCIDENT_COLLECTION: [
            {"era": "fixture_incidents", "filename": INCIDENT_FILE, "url": "fixture"}
        ],
        "ons_fra_boundaries": [
            {"era": "fixture_boundaries", "filename": BOUNDARY_FILE, "url": "fixture"}
        ],
    }
    return Config.model_validate(settings)


def test_convert_raw_reads_incident_files_only(
    two_collection_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script("convert_raw")
    monkeypatch.setattr(script, "load_config", lambda: two_collection_config)
    monkeypatch.setattr(sys, "argv", ["convert_raw.py"])

    assert script.main() == 0

    interim_dir = two_collection_config.paths.resolved("interim_dir")
    written = sorted(path.relative_to(interim_dir) for path in interim_dir.rglob("*.parquet"))
    assert written == [Path("fixture_incidents") / "202324.parquet"]
    assert pq.read_metadata(interim_dir / written[0]).num_rows == REPEATS

    record = json.loads((interim_dir / "conversion_record.json").read_text(encoding="utf-8"))
    assert {sheet["source_file"] for sheet in record["sheets"]} == {INCIDENT_FILE}


def test_build_manifest_reads_incident_files_only(
    two_collection_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script("build_manifest")
    monkeypatch.setattr(script, "load_config", lambda: two_collection_config)

    assert script.main() == 0

    raw_dir = two_collection_config.paths.resolved("raw_dir")
    entries = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [entry["filename"] for entry in entries] == [INCIDENT_FILE]
    assert {entry["collection"] for entry in entries} == {INCIDENT_COLLECTION}


def test_missing_incident_collection_names_what_was_found(
    two_collection_config: Config,
) -> None:
    boundaries = two_collection_config.sources["ons_fra_boundaries"]
    config = two_collection_config.model_copy(
        update={"sources": {"ons_fra_boundaries": boundaries}}
    )
    with pytest.raises(KeyError, match="ons_fra_boundaries"):
        incident_sources(config)
