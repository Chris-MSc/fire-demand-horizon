from __future__ import annotations

import numpy as np
import pytest

from fdh import weather_checks as wc


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
    with pytest.raises(wc.WindowMismatch):
        wc.temperature_ordering(days, days, TASMIN_METHODS, "time: maximum")
