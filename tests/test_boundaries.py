"""Tests for the Rural Urban Classification table and the feature service paging.

The classification fixtures are written through save_table so the saved shape
is the one load_classification reads. The paging test replaces requests.get,
so nothing reaches the network.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from fdh import boundaries as bd


def classification_row(code: str, name: str, settlement_class: str) -> dict:
    return {
        "FRA25CD": code,
        "FRA25NM": name,
        "Rural_Urban_flag": "fixture",
        "RUC21_settlement_class": settlement_class,
        "RUC21_relative_access": "fixture",
        "Proportion_of_population_in_rural_OAs____": 50.0,
        "Proportion_of_population_in_OAs_further_from_a_major_town_or_city____": 25.0,
    }


def test_load_classification_keys_on_fra25cd(tmp_path: Path) -> None:
    rows = [
        classification_row("E31000001", "First", "Majority rural"),
        classification_row("W25000001", "Second", "Urban"),
    ]
    path = tmp_path / "classification.json"
    record = bd.save_table(rows, path)
    assert record["features"] == 2
    assert "crs" not in record

    loaded = bd.load_classification(path)
    assert sorted(loaded) == ["E31000001", "W25000001"]
    assert loaded["E31000001"]["RUC21_settlement_class"] == "Majority rural"


def test_load_classification_names_a_missing_field(tmp_path: Path) -> None:
    row = classification_row("E31000001", "First", "Urban")
    del row["RUC21_relative_access"]
    path = tmp_path / "classification.json"
    bd.save_table([row], path)

    with pytest.raises(bd.BoundaryError, match="RUC21_relative_access"):
        bd.load_classification(path)


def test_load_classification_rejects_an_unexpected_class(tmp_path: Path) -> None:
    rows = [
        classification_row("E31000001", "First", "Urban"),
        classification_row("E31000002", "Second", "Largely rural"),
    ]
    path = tmp_path / "classification.json"
    bd.save_table(rows, path)

    with pytest.raises(bd.BoundaryError, match="Largely rural"):
        bd.load_classification(path)


def crosswalk_row(code: str, name: str) -> bd.CrosswalkRow:
    return bd.CrosswalkRow(
        e_code_territory=code,
        frs_territory=name,
        boundary_code=code,
        boundary_name=name,
        match_basis="e_code",
        name_differs=False,
    )


def test_write_crosswalk_keeps_hand_written_notes(tmp_path: Path) -> None:
    path = tmp_path / "crosswalk.csv"
    rows = [crosswalk_row("E31000001", "Avon"), crosswalk_row("E31000002", "Bedfordshire")]
    bd.write_crosswalk(rows, path)
    text = path.read_text(encoding="utf-8").replace(
        "Avon,e_code,False,", "Avon,e_code,False,uses the 2022 polygon"
    )
    path.write_text(text, encoding="utf-8")

    bd.write_crosswalk(rows, path)

    assert "uses the 2022 polygon" in path.read_text(encoding="utf-8")


def test_write_crosswalk_stops_on_an_orphaned_note(tmp_path: Path) -> None:
    path = tmp_path / "crosswalk.csv"
    bd.write_crosswalk([crosswalk_row("E31000001", "Avon")], path)
    text = path.read_text(encoding="utf-8").replace("Avon,e_code,False,", "Avon,e_code,False,kept")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(bd.BoundaryError, match="E31000001"):
        bd.write_crosswalk([crosswalk_row("E31000002", "Bedfordshire")], path)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.payload


def test_fetch_table_pages_without_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [
        {"features": [{"attributes": {"n": 1}}, {"attributes": {"n": 2}}],
         "exceededTransferLimit": True},
        {"features": [{"attributes": {"n": 3}}]},
    ]
    seen: list[dict[str, list[str]]] = []

    def fake_get(url: str, timeout: int) -> FakeResponse:
        seen.append(parse_qs(urlparse(url).query))
        return FakeResponse(pages[len(seen) - 1])

    monkeypatch.setattr(bd.requests, "get", fake_get)

    assert bd.fetch_table("https://example.invalid/FeatureServer/0") == [
        {"n": 1}, {"n": 2}, {"n": 3}
    ]
    assert [query["resultOffset"] for query in seen] == [["0"], ["2"]]
    assert all(query["returnGeometry"] == ["false"] for query in seen)
    assert all(query["f"] == ["json"] for query in seen)
