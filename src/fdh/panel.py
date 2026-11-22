"""Daily incident counts per service and incident type.

The converted files hold one row per incident with every column as a published
string. This module turns them into a dense daily series: one count for every
combination of calendar day, territory and incident type across the full
recorded span.

Density is the point. An incident that did not happen leaves no row behind, so
a count built by grouping alone carries gaps where the value is zero. Chimney
fires in a small territory are zero on most days of the year. Those days are
observations and are written as zero.

Days inside a known reporting gap are written as null rather than zero, since
the published guidance records them as not submitted rather than as nothing
happening. The two cannot be told apart from the data.

Three properties of the published file are checked rather than assumed: that
the date built from CALENDAR_YEAR, MONTH_CODE and DAY is a real date, that the
weekday it falls on matches the published DAY_OF_WEEK, and that the calendar
month sits inside the published FINANCIAL_YEAR. A single row failing any of
them stops the build.

Reporting gaps are known only where the guidance declares them. Days on which
a territory recorded nothing at all are reported by `quiet_days` so that an
undeclared gap is seen rather than modelled as a run of zeros.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pyarrow.parquet as pq

# Index order matches datetime.date.weekday(), Monday first. Derived rather
# than formatted, so the check does not depend on the runtime locale.
WEEKDAY_NAMES: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Four-digit years inside the published FINANCIAL_YEAR string, whatever
# separator it uses.
YEAR_IN_TEXT = re.compile(r"\d{4}")

READ_COLUMNS: tuple[str, ...] = (
    "FRS_TERRITORY",
    "E_CODE_TERRITORY",
    "FINANCIAL_YEAR",
    "CALENDAR_YEAR",
    "MONTH_CODE",
    "DAY",
    "DAY_OF_WEEK",
    "FRIS_INCIDENT_TYPE",
)


class PanelError(RuntimeError):
    """Raised when the converted files cannot produce a dense daily panel."""


@dataclass
class Rejected:
    """A row that could not be placed on the calendar, kept with its reason."""

    source: str
    row: int
    reason: str
    values: dict[str, str]


@dataclass
class ReadResult:
    """Counts keyed on day, territory and type, with what was checked alongside."""

    counts: dict[tuple[date, str, str], int] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    years_present: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    types_seen: set[str] = field(default_factory=set)
    rows_read: int = 0
    # The lists hold the first examples only; the counts hold every case.
    rejected: list[Rejected] = field(default_factory=list)
    weekday_mismatches: list[Rejected] = field(default_factory=list)
    financial_year_mismatches: list[Rejected] = field(default_factory=list)
    rejected_count: int = 0
    weekday_mismatch_count: int = 0
    financial_year_mismatch_count: int = 0

    @property
    def first_day(self) -> date:
        return min(key[0] for key in self.counts)

    @property
    def last_day(self) -> date:
        return max(key[0] for key in self.counts)


def financial_year_start(day: date) -> int:
    """Return the calendar year the financial year containing `day` begins in."""
    return day.year if day.month >= 4 else day.year - 1


def build_date(year: str, month_code: str, day: str) -> date:
    """Return the calendar date, or raise ValueError naming what was wrong."""
    return date(int(year), int(month_code), int(day))


def days_between(first: date, last: date) -> list[date]:
    if last < first:
        raise PanelError(f"span ends before it starts: {first} to {last}")
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def read_counts(interim_dir: Path, eras: list[str], limit_rejected: int = 50) -> ReadResult:
    """Aggregate the converted incident files into counts per day, territory and type.

    Only the directories named in `eras` are read, so a directory left behind
    by an earlier conversion under another name is never counted. A financial
    year held in more than one file stops the read, since its incidents would
    be counted twice.

    Rows that do not yield a real date, and rows whose date disagrees with the
    published weekday or financial year, are all counted, with the first
    `limit_rejected` of each kept as examples. The read itself does not stop
    on them, so the whole extent is known; `check_dates` stops the build.
    """
    files = [path for era in eras for path in sorted((interim_dir / era).glob("*.parquet"))]
    if not files:
        raise PanelError(
            f"no converted incident files under {interim_dir} for {eras}. "
            f"Run scripts/convert_raw.py first."
        )

    result = ReadResult()
    held_in: dict[str, list[str]] = defaultdict(list)
    for path in files:
        source = f"{path.parent.name}/{path.name}"
        table = pq.read_table(path, columns=list(READ_COLUMNS))
        columns = {name: table.column(name).to_pylist() for name in READ_COLUMNS}
        for year in set(columns["FINANCIAL_YEAR"]):
            held_in[year].append(source)

        for index in range(table.num_rows):
            result.rows_read += 1
            values = {name: columns[name][index] for name in READ_COLUMNS}
            code = values["E_CODE_TERRITORY"]

            try:
                day = build_date(
                    values["CALENDAR_YEAR"], values["MONTH_CODE"], values["DAY"]
                )
            except (TypeError, ValueError) as error:
                result.rejected_count += 1
                if len(result.rejected) < limit_rejected:
                    result.rejected.append(
                        Rejected(source, index + 2, str(error), values)
                    )
                continue

            published_weekday = values["DAY_OF_WEEK"]
            if published_weekday != WEEKDAY_NAMES[day.weekday()]:
                result.weekday_mismatch_count += 1
                if len(result.weekday_mismatches) < limit_rejected:
                    result.weekday_mismatches.append(
                        Rejected(
                            source,
                            index + 2,
                            f"{day.isoformat()} is a "
                            f"{WEEKDAY_NAMES[day.weekday()]}, published as "
                            f"{published_weekday}",
                            values,
                        )
                    )

            years = YEAR_IN_TEXT.findall(values["FINANCIAL_YEAR"] or "")
            if str(financial_year_start(day)) not in years:
                result.financial_year_mismatch_count += 1
                if len(result.financial_year_mismatches) < limit_rejected:
                    result.financial_year_mismatches.append(
                        Rejected(
                            source,
                            index + 2,
                            f"{day.isoformat()} sits in financial year "
                            f"{financial_year_start(day)}, published as "
                            f"{values['FINANCIAL_YEAR']}",
                            values,
                        )
                    )

            key = (day, code, values["FRIS_INCIDENT_TYPE"])
            result.counts[key] = result.counts.get(key, 0) + 1
            result.names[code] = values["FRS_TERRITORY"]
            result.years_present[code].add(values["FINANCIAL_YEAR"])
            result.types_seen.add(values["FRIS_INCIDENT_TYPE"])

    twice = {year: sources for year, sources in sorted(held_in.items()) if len(sources) > 1}
    if twice:
        raise PanelError(
            "a financial year is held in more than one converted file, so its "
            "incidents would be counted twice.\n"
            "  expected: each financial year in one file\n"
            f"  found:    {twice}"
        )
    if not result.counts:
        raise PanelError(f"{len(files)} files read and no rows placed on the calendar")
    return result


def check_dates(result: ReadResult) -> None:
    """Stop unless every row gave a real date agreeing with the published fields.

    One failing row among millions stops the build. The published weekday
    restates the date, so agreement on every row is what shows the date
    construction to be right, and a single disagreement means the construction
    is wrong for some part of the file rather than that one row is odd. A
    genuine publisher error is handled by declaring it in configuration with
    its source, not by tolerating a count here.
    """
    kinds = (
        ("rows with no real date", result.rejected_count, result.rejected),
        ("weekday disagreements", result.weekday_mismatch_count, result.weekday_mismatches),
        (
            "financial year disagreements",
            result.financial_year_mismatch_count,
            result.financial_year_mismatches,
        ),
    )
    failed = [(label, count, examples) for label, count, examples in kinds if count]
    if not failed:
        return
    lines = [
        "dates built from CALENDAR_YEAR, MONTH_CODE and DAY do not agree with the "
        "published fields.",
        f"  expected: 0 of {result.rows_read:,} rows failing each check",
    ]
    for label, count, examples in failed:
        lines.append(f"  found:    {label}: {count:,}")
        for item in examples[:3]:
            lines.append(f"            {item.source} row {item.row}: {item.reason}")
    raise PanelError("\n".join(lines))


def check_territories_complete(result: ReadResult) -> list[str]:
    """Return territories absent from any financial year the others appear in.

    A territory missing from a year would be given invented zeros for that year
    by the reindex, so the condition stops the build rather than being filled.
    """
    every_year = set().union(*result.years_present.values())
    return sorted(
        code
        for code, years in result.years_present.items()
        if years != every_year
    )


def gap_days(gaps: list[tuple[str, date, date]]) -> set[tuple[date, str]]:
    """Expand reporting gaps into the day and territory pairs they cover."""
    covered: set[tuple[date, str]] = set()
    for code, start, end in gaps:
        for day in days_between(start, end):
            covered.add((day, code))
    return covered


def build_rows(
    result: ReadResult,
    types: list[str],
    gaps: list[tuple[str, date, date]] | None = None,
    first_day: date | None = None,
    last_day: date | None = None,
) -> tuple[list[dict], dict]:
    """Return one row per day, territory and type, with a record of what was built.

    Counts are zero where nothing was recorded and null where the day falls
    inside a reporting gap for that territory. The span must sit inside the
    recorded span: days beyond the last recorded day would be filled with
    zeros that nothing observed.
    """
    check_dates(result)
    missing = check_territories_complete(result)
    if missing:
        labelled = [f"{code} ({result.names.get(code, '?')})" for code in missing]
        raise PanelError(
            "territories do not appear in every financial year, so a dense "
            "panel would invent zeros for years they are absent from.\n"
            f"  affected: {labelled}"
        )

    unknown = sorted(set(types) - result.types_seen)
    if unknown:
        raise PanelError(
            "incident types configured but never recorded in the data.\n"
            f"  configured but absent: {unknown}\n"
            f"  recorded:              {sorted(result.types_seen)}"
        )

    first = first_day or result.first_day
    last = last_day or result.last_day
    if first < result.first_day or last > result.last_day:
        raise PanelError(
            "the panel span reaches outside the recorded span, so the days "
            "beyond it would be written as zero with nothing observed.\n"
            f"  expected: within {result.first_day} to {result.last_day}\n"
            f"  found:    {first} to {last}"
        )
    calendar = days_between(first, last)
    codes = sorted(result.names)
    covered = gap_days(gaps or [])

    rows: list[dict] = []
    zeros = 0
    nulls = 0
    for day in calendar:
        for code in codes:
            in_gap = (day, code) in covered
            for incident_type in types:
                count = result.counts.get((day, code, incident_type), 0)
                if in_gap:
                    rows.append(
                        {
                            "incident_date": day,
                            "e_code_territory": code,
                            "frs_territory": result.names[code],
                            "incident_type": incident_type,
                            "count": None,
                            "reporting_gap": True,
                        }
                    )
                    nulls += 1
                    continue
                if count == 0:
                    zeros += 1
                rows.append(
                    {
                        "incident_date": day,
                        "e_code_territory": code,
                        "frs_territory": result.names[code],
                        "incident_type": incident_type,
                        "count": count,
                        "reporting_gap": False,
                    }
                )

    # Each incident is attributed to the first reason that applies, so the
    # four counts partition the rows read and sum back to it.
    wanted = set(types)
    placed = outside_span = outside_types = inside_gap = 0
    for (day, code, incident_type), count in result.counts.items():
        if not (first <= day <= last):
            outside_span += count
        elif incident_type not in wanted:
            outside_types += count
        elif (day, code) in covered:
            inside_gap += count
        else:
            placed += count

    record = {
        "rows_read": result.rows_read,
        "incidents_placed": placed,
        "incidents_outside_span": outside_span,
        "incidents_outside_requested_types": outside_types,
        "incidents_inside_reporting_gap": inside_gap,
        "rows_written": len(rows),
        "first_day": first.isoformat(),
        "last_day": last.isoformat(),
        "days": len(calendar),
        "territories": len(codes),
        "incident_types": len(types),
        "zero_days": zeros,
        "zero_share": round(zeros / len(rows), 6) if rows else 0.0,
        "reporting_gap_cells": nulls,
        "rejected_rows": result.rejected_count,
        "weekday_mismatches": result.weekday_mismatch_count,
        "financial_year_mismatches": result.financial_year_mismatch_count,
    }
    return rows, record


def quiet_days(
    result: ReadResult,
    first_day: date,
    last_day: date,
    gaps: list[tuple[str, date, date]] | None = None,
) -> list[dict]:
    """Return territory-days with no incidents of any type, with the territory's median day.

    A whole service recording nothing is more likely a day never submitted
    than a day on which nothing happened, and the data cannot tell the two
    apart.

    Days inside a declared reporting gap are left out of the result and of the
    median. Nothing is changed in the panel: a day returned here becomes null
    only by being declared as a reporting gap with its source.
    """
    covered = gap_days(gaps or [])
    totals: dict[tuple[date, str], int] = defaultdict(int)
    for (day, code, _), count in result.counts.items():
        totals[(day, code)] += count

    calendar = days_between(first_day, last_day)
    found: list[dict] = []
    for code in sorted(result.names):
        observed = [
            (day, totals.get((day, code), 0)) for day in calendar if (day, code) not in covered
        ]
        if not observed:
            continue
        median = statistics.median(total for _, total in observed)
        for day, total in observed:
            if total == 0:
                found.append(
                    {
                        "incident_date": day.isoformat(),
                        "e_code_territory": code,
                        "frs_territory": result.names[code],
                        "incidents": total,
                        "territory_median": median,
                    }
                )
    return sorted(found, key=lambda row: (row["incident_date"], row["e_code_territory"]))


def zero_shares(rows: list[dict]) -> list[dict]:
    """Return the share of days with no incidents for each territory and type.

    Cells inside a reporting gap are left out of both numerator and denominator.
    The distribution across series is what the minimum non-zero day count for
    scoring eligibility is set from.
    """
    totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    names: dict[str, str] = {}
    for row in rows:
        if row["reporting_gap"]:
            continue
        key = (row["e_code_territory"], row["incident_type"])
        totals[key][0] += 1
        if row["count"] == 0:
            totals[key][1] += 1
        names[row["e_code_territory"]] = row["frs_territory"]

    summary = []
    for (code, incident_type), (days, zero_days) in sorted(totals.items()):
        summary.append(
            {
                "e_code_territory": code,
                "frs_territory": names[code],
                "incident_type": incident_type,
                "days": days,
                "zero_days": zero_days,
                "non_zero_days": days - zero_days,
                "zero_share": round(zero_days / days, 6) if days else 0.0,
            }
        )
    return summary
