from __future__ import annotations

import zipfile
from pathlib import Path

from fdh.ods import iter_rows

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


def read(tmp_path: Path, rows: str) -> list[list[str]]:
    path = tmp_path / "fixture.ods"
    tables = f'<table:table table:name="202324">{rows}</table:table>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet")
        archive.writestr("content.xml", CONTENT_TEMPLATE.format(tables=tables))
    return [cells for _, cells in iter_rows(path, sheet="202324")]


def test_row_repeats_are_expanded(tmp_path: Path) -> None:
    rows = read(
        tmp_path,
        row(cell("FRS_TERRITORY") + cell("FRIS_INCIDENT_TYPE"))
        + row(cell("Suffolk") + cell("Secondary Fires"), repeat=37)
        + row(cell("Suffolk") + cell("Chimney Fires")),
    )
    assert len(rows) == 39, "header plus 37 repeated rows plus one"
    assert sum(1 for cells in rows if cells[1] == "Secondary Fires") == 37


def test_empty_rows_are_not_emitted(tmp_path: Path) -> None:
    rows = read(tmp_path, row(cell("A") + cell("B")) + row(cell("", repeat=9), repeat=500_000))
    assert rows == [["A", "B"]]


def test_column_repeats_preserve_position(tmp_path: Path) -> None:
    rows = read(tmp_path, row(cell("first") + cell("", repeat=3) + cell("fifth")))
    assert rows == [["first", "", "", "", "fifth"]]


def test_trailing_blanks_are_trimmed(tmp_path: Path) -> None:
    rows = read(tmp_path, row(cell("only") + cell("", repeat=12)))
    assert rows == [["only"]]
