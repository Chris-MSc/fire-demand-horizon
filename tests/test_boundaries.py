"""Tests for the crosswalk's hand-written notes and the classification classes."""

from __future__ import annotations

from pathlib import Path

import pytest

from fdh import boundaries as bd


def classification_row(code: str, settlement_class: str) -> dict:
    return {
        "FRA25CD": code,
        "FRA25NM": "fixture",
        "Rural_Urban_flag": "fixture",
        "RUC21_settlement_class": settlement_class,
        "RUC21_relative_access": "fixture",
        "Proportion_of_population_in_rural_OAs____": 50.0,
        "Proportion_of_population_in_OAs_further_from_a_major_town_or_city____": 25.0,
    }


def crosswalk_row(code: str, name: str) -> bd.CrosswalkRow:
    return bd.CrosswalkRow(
        e_code_territory=code,
        frs_territory=name,
        boundary_code=code,
        boundary_name=name,
        match_basis="e_code",
        name_differs=False,
    )


def with_note(path: Path, note: str) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("Avon,e_code,False,", f"Avon,e_code,False,{note}"), "utf-8")


def test_a_settlement_class_outside_the_published_four_is_refused(tmp_path: Path) -> None:
    rows = [
        classification_row("E31000001", "Urban"),
        classification_row("E31000002", "Largely rural"),
    ]
    path = tmp_path / "classification.json"
    bd.save_table(rows, path)

    with pytest.raises(bd.BoundaryError):
        bd.load_classification(path)


def test_rewriting_the_crosswalk_keeps_hand_written_notes(tmp_path: Path) -> None:
    path = tmp_path / "crosswalk.csv"
    rows = [crosswalk_row("E31000001", "Avon"), crosswalk_row("E31000002", "Bedfordshire")]
    bd.write_crosswalk(rows, path)
    with_note(path, "uses the 2022 polygon")

    bd.write_crosswalk(rows, path)

    assert "uses the 2022 polygon" in path.read_text(encoding="utf-8")


def test_a_note_on_a_territory_the_crosswalk_no_longer_holds_stops_the_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "crosswalk.csv"
    bd.write_crosswalk([crosswalk_row("E31000001", "Avon")], path)
    with_note(path, "kept")

    with pytest.raises(bd.BoundaryError):
        bd.write_crosswalk([crosswalk_row("E31000002", "Bedfordshire")], path)
