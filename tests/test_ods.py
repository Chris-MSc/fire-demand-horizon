from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from fdh.ods import OdsStructureError, iter_rows, read_config_sheet, sheet_names

CONTENT_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
  <office:body><office:spreadsheet>{tables}</office:spreadsheet></office:body>
</office:document-content>
"""


def cell(value: str, repeat: int = 1) -> str:
    attr = f' table:number-columns-repeated="{repeat}"' if repeat > 1 else ""
    if value == "":
        return f"<table:table-cell{attr}/>"
    return f"<table:table-cell{attr}><text:p>{value}</text:p></table:table-cell>"


def row(cells: str, repeat: int = 1) -> str:
    attr = f' table:number-rows-repeated="{repeat}"' if repeat > 1 else ""
    return f"<table:table-row{attr}>{cells}</table:table-row>"


def make_ods(path: Path, tables: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet")
        archive.writestr("content.xml", CONTENT_TEMPLATE.format(tables=tables))
    return path


def test_row_repeats_are_expanded(tmp_path: Path) -> None:
    tables = (
        '<table:table table:name="202324">'
        + row(cell("FRS_TERRITORY") + cell("FRIS_INCIDENT_TYPE"))
        + row(cell("Suffolk") + cell("Secondary Fires"), repeat=37)
        + row(cell("Suffolk") + cell("Chimney Fires"))
        + "</table:table>"
    )
    path = make_ods(tmp_path / "repeat.ods", tables)
    rows = list(iter_rows(path, sheet="202324"))
    assert len(rows) == 39, "header plus 37 repeated rows plus one"
    assert sum(1 for _, cells in rows if cells[1] == "Secondary Fires") == 37


def test_empty_rows_are_not_emitted(tmp_path: Path) -> None:
    tables = (
        '<table:table table:name="202324">'
        + row(cell("A") + cell("B"))
        + row(cell("", repeat=9), repeat=500_000)
        + "</table:table>"
    )
    path = make_ods(tmp_path / "padding.ods", tables)
    assert len(list(iter_rows(path, sheet="202324"))) == 1


def test_absurd_repeat_raises(tmp_path: Path) -> None:
    tables = (
        '<table:table table:name="202324">'
        + row(cell("Suffolk") + cell("Dwellings"), repeat=5_000_000)
        + "</table:table>"
    )
    path = make_ods(tmp_path / "absurd.ods", tables)
    with pytest.raises(OdsStructureError):
        list(iter_rows(path, sheet="202324"))


def test_column_repeats_preserve_position(tmp_path: Path) -> None:
    tables = (
        '<table:table table:name="202324">'
        + row(cell("first") + cell("", repeat=3) + cell("fifth"))
        + "</table:table>"
    )
    path = make_ods(tmp_path / "columns.ods", tables)
    _, cells = next(iter(iter_rows(path, sheet="202324")))
    assert cells == ["first", "", "", "", "fifth"]


def test_trailing_blanks_are_trimmed(tmp_path: Path) -> None:
    tables = (
        '<table:table table:name="202324">'
        + row(cell("only") + cell("", repeat=12))
        + "</table:table>"
    )
    path = make_ods(tmp_path / "trailing.ods", tables)
    _, cells = next(iter(iter_rows(path, sheet="202324")))
    assert cells == ["only"]


def test_sheet_names_and_config_sheet(tmp_path: Path) -> None:
    tables = (
        '<table:table table:name="Cover_sheet">' + row(cell("ignored")) + "</table:table>"
        '<table:table table:name="202324">' + row(cell("A")) + "</table:table>"
        '<table:table table:name="config">'
        + row(cell("Pubdate") + cell("22 July 2026"))
        + row(cell("Datacut date") + cell("12-May-26"))
        + "</table:table>"
    )
    path = make_ods(tmp_path / "named.ods", tables)
    assert sheet_names(path) == ["Cover_sheet", "202324", "config"]
    values = read_config_sheet(path, ("config",))
    assert values["Pubdate"] == "22 July 2026"
    assert values["Datacut date"] == "12-May-26"


def test_missing_config_sheet_raises(tmp_path: Path) -> None:
    tables = '<table:table table:name="202324">' + row(cell("A")) + "</table:table>"
    path = make_ods(tmp_path / "noconfig.ods", tables)
    with pytest.raises(OdsStructureError):
        read_config_sheet(path, ("config",))