from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from fdh import weather as wx
from fdh import weather_checks as wc
from fdh.weather import ArchiveFile


def test_ensure_token_prompts_only_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []
    monkeypatch.setattr(wx.getpass, "getpass", lambda text: prompts.append(text) or " t0k ")

    monkeypatch.delenv("CEDA_ACCESS_TOKEN", raising=False)
    assert wx.ensure_token() is True
    assert wx.token() == "t0k"

    assert wx.ensure_token() is True
    assert len(prompts) == 1, "a token already set is not asked for again"


def month(year: int, number: int, last_day: int) -> ArchiveFile:
    start, end = date(year, number, 1), date(year, number, last_day)
    return ArchiveFile(name=f"{start}", url="", size=1, start=start, end=end)


def window(variable: str, start_hours: list[int], cell_methods: str, comment=None) -> dict:
    return {
        "variable": variable,
        "cell_methods": cell_methods,
        "comment": comment,
        "start_hours": start_hours,
        "span_hours": [24.0],
    }


def test_window_rows_separate_silence_from_disagreement() -> None:
    rows = wc.window_rows(
        [
            window("rainfall", [9], "time: sum"),
            window("tasmax", [0], "time: maximum", "observed 0900 to 0900"),
            window("tasmin", [9], "time: minimum (interval: 09:00 to 09:00)"),
        ]
    )
    assert [row["agreement"] for row in rows] == ["not stated", "disagree", "agree"]


def test_window_rows_ignore_0900_inside_a_longer_number() -> None:
    (row,) = wc.window_rows([window("rainfall", [0], "time: sum", "reference 20090001")])
    assert row["agreement"] == "not stated"


def test_coverage_rows_name_the_month_a_variable_lacks() -> None:
    rows = wc.coverage_rows(
        {
            "rainfall": [month(2010, 4, 30), month(2010, 5, 31)],
            "tasmax": [month(2010, 4, 30)],
        }
    )
    missing = {row["variable"]: row["missing"] for row in rows}
    assert missing == {"rainfall": [], "tasmax": [(2010, 5)]}


def test_contiguity_rows_count_gaps_and_overlaps() -> None:
    gap = [month(2010, 4, 30), month(2010, 6, 30)]
    overlap = [month(2010, 4, 30), ArchiveFile("x", "", 1, date(2010, 4, 30), date(2010, 5, 31))]
    rows = {row["variable"]: row for row in wc.contiguity_rows({"gap": gap, "overlap": overlap})}
    assert (rows["gap"]["gaps"], rows["gap"]["overlaps"]) == (1, 0)
    assert (rows["overlap"]["gaps"], rows["overlap"]["overlaps"]) == (0, 1)


TASMIN_METHODS = f"time: minimum (comment: observing period is {wc.TASMIN_WINDOW})"
TASMAX_METHODS = f"time: maximum (comment: observing period is {wc.TASMAX_WINDOW})"


def test_temperature_ordering_compares_tasmin_of_the_next_day() -> None:
    # Two days, two cells. Day 0 is warm then a cold night follows; cell 1 is masked.
    tasmax = np.ma.masked_array([[10.0, 5.0], [4.0, 5.0]], mask=[[0, 1], [0, 1]])
    tasmin = np.ma.masked_array([[12.0, 1.0], [11.0, 1.0]], mask=[[0, 1], [0, 1]])
    result = wc.temperature_ordering(tasmin, tasmax, TASMIN_METHODS, TASMAX_METHODS)

    # Matched pair: tasmin[1] = 11 against tasmax[0] = 10, one failure of 1 C.
    assert (result["cell_days"], result["failing"]) == (1, 1)
    assert result["max_excess"] == pytest.approx(1.0)
    # Same day: 12 > 10 and 11 > 4, both fail.
    assert (result["same_day_cell_days"], result["same_day_failing"]) == (2, 2)


def test_temperature_ordering_stops_on_another_stated_window() -> None:
    days = np.ma.masked_array(np.zeros((2, 1)))
    with pytest.raises(wc.WindowMismatch, match="tasmax"):
        wc.temperature_ordering(days, days, TASMIN_METHODS, "time: maximum")


def test_expected_months_counts_both_ends() -> None:
    assert wc.expected_months(date(2010, 4, 1), date(2025, 12, 31)) == 189
