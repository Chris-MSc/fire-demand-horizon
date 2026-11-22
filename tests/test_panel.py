from __future__ import annotations

from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fdh.contract import EXPECTED_COLUMNS
from fdh.panel import (
    PanelError,
    build_rows,
    financial_year_start,
    quiet_days,
    read_counts,
    zero_shares,
)

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
AVON = "E31000001"
GAP = [(AVON, date(2010, 4, 2), date(2010, 4, 2))]


def incident(
    day: date,
    code: str = AVON,
    name: str = "Avon",
    incident_type: str = "Chimney Fires",
    financial_year: str | None = None,
    day_of_week: str | None = None,
) -> dict[str, str]:
    start = financial_year_start(day)
    return {
        "FRS_TERRITORY": name,
        "E_CODE_TERRITORY": code,
        "FINANCIAL_YEAR": financial_year or f"{start}/{str(start + 1)[2:]}",
        "CALENDAR_YEAR": str(day.year),
        "MONTH": day.strftime("%B"),
        "MONTH_CODE": str(day.month),
        "DAY": str(day.day),
        "DAY_OF_WEEK": day_of_week or WEEKDAYS[day.weekday()],
        "FRIS_INCIDENT_TYPE": incident_type,
    }


def write_era(directory: Path, era: str, rows: list[dict[str, str]]) -> None:
    target = directory / era
    target.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([(name, pa.string()) for name in EXPECTED_COLUMNS])
    table = pa.Table.from_arrays(
        [pa.array([row[name] for row in rows], type=pa.string()) for name in EXPECTED_COLUMNS],
        schema=schema,
    )
    pq.write_table(table, target / f"{era}.parquet")


def read(tmp_path: Path, rows: list[dict[str, str]]):
    write_era(tmp_path, "era_one", rows)
    return read_counts(tmp_path, ["era_one"])


def test_days_with_no_incidents_are_zero_for_every_territory_and_type(tmp_path):
    first, third = date(2010, 4, 1), date(2010, 4, 3)
    result = read(
        tmp_path,
        [
            incident(first),
            incident(first),
            incident(third),
            incident(first, code="E31000002", name="Bedfordshire", incident_type="Dwellings"),
            incident(third, code="E31000002", name="Bedfordshire", incident_type="Dwellings"),
        ],
    )

    rows, _ = build_rows(result, ["Chimney Fires", "Dwellings"])

    assert len(rows) == 12, "three days, two territories, two types"
    counts = {
        (row["incident_date"].day, row["e_code_territory"], row["incident_type"]): row["count"]
        for row in rows
    }
    assert [counts[(day, AVON, "Chimney Fires")] for day in (1, 2, 3)] == [2, 0, 1]
    assert [counts[(day, AVON, "Dwellings")] for day in (1, 2, 3)] == [0, 0, 0]


def test_a_day_inside_a_reporting_gap_is_null_and_left_out_of_the_zero_shares(tmp_path):
    result = read(tmp_path, [incident(date(2010, 4, 1)), incident(date(2010, 4, 3))])

    rows, _ = build_rows(result, ["Chimney Fires"], gaps=GAP)

    assert [row["count"] for row in rows] == [1, None, 1]
    assert [row["reporting_gap"] for row in rows] == [False, True, False]
    (series,) = zero_shares(rows)
    assert (series["days"], series["zero_days"]) == (2, 0)


def test_a_single_weekday_that_disagrees_stops_the_build(tmp_path):
    day = date(2010, 4, 1)
    result = read(tmp_path, [incident(day), incident(day, day_of_week="Sun")])

    with pytest.raises(PanelError):
        build_rows(result, ["Chimney Fires"])


def test_a_row_with_no_real_date_stops_the_build(tmp_path):
    bad = incident(date(2010, 4, 1))
    bad["DAY"], bad["MONTH_CODE"] = "31", "2"
    result = read(tmp_path, [bad, incident(date(2010, 4, 1))])

    with pytest.raises(PanelError):
        build_rows(result, ["Chimney Fires"])


def test_a_date_outside_its_published_financial_year_stops_the_build(tmp_path):
    result = read(tmp_path, [incident(date(2010, 4, 1), financial_year="2009/10")])

    with pytest.raises(PanelError):
        build_rows(result, ["Chimney Fires"])


def test_a_financial_year_held_in_two_files_stops_the_read(tmp_path):
    write_era(tmp_path, "era_one", [incident(date(2010, 4, 1))])
    write_era(tmp_path, "era_two", [incident(date(2010, 5, 1))])

    with pytest.raises(PanelError):
        read_counts(tmp_path, ["era_one", "era_two"])


def test_a_territory_absent_from_a_year_stops_the_build(tmp_path):
    result = read(
        tmp_path,
        [
            incident(date(2010, 4, 1)),
            incident(date(2011, 4, 1)),
            incident(date(2010, 4, 1), code="E31000002", name="Bedfordshire"),
        ],
    )

    with pytest.raises(PanelError):
        build_rows(result, ["Chimney Fires"])


def test_a_span_past_the_last_recorded_day_stops_the_build(tmp_path):
    result = read(tmp_path, [incident(date(2010, 4, 1)), incident(date(2010, 4, 3))])

    with pytest.raises(PanelError):
        build_rows(result, ["Chimney Fires"], last_day=date(2010, 4, 4))


def test_the_calendar_is_cut_to_the_given_span(tmp_path):
    result = read(tmp_path, [incident(date(2010, 4, day)) for day in (1, 2, 3, 5, 5)])

    rows, record = build_rows(
        result, ["Chimney Fires"], first_day=date(2010, 4, 2), last_day=date(2010, 4, 4)
    )

    assert [row["incident_date"].day for row in rows] == [2, 3, 4]
    assert [row["count"] for row in rows] == [1, 1, 0]
    assert record["incidents_outside_span"] == 3


def test_quiet_days_are_days_with_nothing_of_any_type_outside_a_declared_gap(tmp_path):
    # 2 April is inside the declared gap. 3 April has a dwelling fire and no
    # chimney fire, so something was recorded. 4 April has nothing of any type.
    result = read(
        tmp_path,
        [
            incident(date(2010, 4, 1)),
            incident(date(2010, 4, 3), incident_type="Dwellings"),
            incident(date(2010, 4, 5)),
        ],
    )

    found = quiet_days(result, date(2010, 4, 1), date(2010, 4, 5), gaps=GAP)

    assert [(row["incident_date"], row["incidents"]) for row in found] == [("2010-04-04", 0)]
