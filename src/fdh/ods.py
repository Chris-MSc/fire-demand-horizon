"""Streaming reader for the published ODS files.

Rows carrying table:number-rows-repeated are emitted once per repeat. In the
2023-24 to present file, 1.75 million of 1.85 million data rows are stored
that way.

content.xml is parsed incrementally rather than loaded whole. Uncompressed it
is 363 MB in the smallest of the three era files.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path

from lxml import etree

TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
T = f"{{{TABLE_NS}}}"

# Trailing padding rows claim repeat counts in the millions. Empty rows are
# never emitted; this cap bounds allocation on a malformed file.
MAX_SANE_REPEAT = 1_000_000


class OdsStructureError(RuntimeError):
    """Raised when a file departs from the structure in fdh.contract."""


def _cell_text(cell: etree._Element) -> str:
    return "".join(cell.itertext())


def _row_cells(row: etree._Element) -> list[str]:
    """Return a row's cells, expanding column repeats, trailing blanks trimmed."""
    cells: list[str] = []
    pending_blanks = 0
    for cell in row:
        tag = cell.tag
        if tag not in (f"{T}table-cell", f"{T}covered-table-cell"):
            continue
        repeat = int(cell.get(f"{T}number-columns-repeated", "1"))
        text = _cell_text(cell)
        if text == "":
            pending_blanks += repeat
            continue
        if pending_blanks:
            cells.extend([""] * pending_blanks)
            pending_blanks = 0
        cells.extend([text] * repeat)
    return cells


def sheet_names(path: Path) -> list[str]:
    names: list[str] = []
    with zipfile.ZipFile(path) as archive, archive.open("content.xml") as stream:
        for event, element in etree.iterparse(stream, events=("start",)):
            if element.tag == f"{T}table":
                names.append(element.get(f"{T}name", ""))
    return names


def iter_rows(path: Path, sheet: str | None = None) -> Iterator[tuple[str, list[str]]]:
    """Yield (sheet_name, cells) for every non-empty row, in document order.

    Row repeats are expanded. Header rows are returned like any other row.
    """
    if not path.exists():
        raise FileNotFoundError(f"ODS file not found: {path}")
    current: str | None = None
    with zipfile.ZipFile(path) as archive, archive.open("content.xml") as stream:
        for event, element in etree.iterparse(stream, events=("start", "end")):
            if event == "start" and element.tag == f"{T}table":
                current = element.get(f"{T}name", "")
                continue
            if event != "end":
                continue
            if element.tag == f"{T}table-row":
                if sheet is None or current == sheet:
                    cells = _row_cells(element)
                    if cells:
                        repeat = int(element.get(f"{T}number-rows-repeated", "1"))
                        if repeat > MAX_SANE_REPEAT:
                            raise OdsStructureError(
                                f"{path.name}, sheet {current}: row repeat of {repeat} "
                                f"exceeds the sane limit of {MAX_SANE_REPEAT}"
                            )
                        for _ in range(repeat):
                            yield current or "", cells
                _drop(element)
            elif element.tag == f"{T}table":
                _drop(element)


def _drop(element: etree._Element) -> None:
    element.clear()
    parent = element.getparent()
    if parent is not None:
        while element.getprevious() is not None:
            del parent[0]


def read_config_sheet(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    """Return the key and value pairs of a file's own config sheet.

    The sheet holds the publication date and data cut date of that release.
    """
    present = set(sheet_names(path))
    for candidate in names:
        if candidate in present:
            values: dict[str, str] = {}
            for _, cells in iter_rows(path, sheet=candidate):
                if len(cells) >= 2 and cells[0]:
                    values[cells[0].strip()] = cells[1].strip()
            return values
    raise OdsStructureError(
        f"{path.name}: no config sheet found. Expected one of {names}, "
        f"found {sorted(present)}"
    )