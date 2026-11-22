"""Configuration loading.

Open decisions are null in config.yaml. Reading one through `require` raises
instead of returning a default, so a stage that depends on an undecided value
stops rather than producing a plausible answer from an invented one.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from fdh.contract import SETTLEMENT_CLASS, SETTLEMENT_CLASSES


class UndecidedError(RuntimeError):
    """Raised when code needs a configuration value that has not been decided."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Paths(BaseModel):
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path

    def resolved(self, name: str) -> Path:
        value = getattr(self, name)
        return (repo_root() / value).resolve() if not value.is_absolute() else value


class Source(BaseModel):
    era: str
    filename: str
    url: str
    sha256: str | None = None


class Panel(BaseModel):
    services: list[str] | None = None
    rural_urban_classes: list[str] | None = None
    start_date: date | None = None
    end_date: date | None = None

    @field_validator("rural_urban_classes")
    @classmethod
    def check_classes(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        unknown = [name for name in value if name not in SETTLEMENT_CLASSES]
        if unknown:
            raise ValueError(
                f"panel.rural_urban_classes holds values that are not "
                f"{SETTLEMENT_CLASS} classes.\n"
                f"  expected any of: {sorted(SETTLEMENT_CLASSES)}\n"
                f"  found:           {unknown}"
            )
        if len(set(value)) != len(value):
            raise ValueError(f"panel.rural_urban_classes repeats a class: {value}")
        return value


class IncidentTypes(BaseModel):
    modelled_types: list[str]
    mapping_file: Path | None = None
    crosswalk_file: Path | None = None


class Horizon(BaseModel):
    min_days: int = Field(gt=0)
    max_days: int = Field(gt=0)
    step_days: int = Field(gt=0)
    mase_threshold: float = Field(gt=0)

    @model_validator(mode="after")
    def check_grid(self) -> Horizon:
        if self.max_days < self.min_days:
            raise ValueError("horizon.max_days is below horizon.min_days")
        span = self.max_days - self.min_days
        if span % self.step_days:
            raise ValueError(
                f"horizon grid is not whole: {self.min_days} to {self.max_days} "
                f"does not divide by step {self.step_days}"
            )
        return self

    @property
    def lead_times(self) -> list[int]:
        return list(range(self.min_days, self.max_days + 1, self.step_days))


class Model(BaseModel):
    seasonal_period: int = Field(gt=0)
    fourier_terms: int | None = None
    information_condition: Literal["restricted", "perfect_foresight"]


class Benchmark(BaseModel):
    primary: str
    robustness: str | None = None


class ReportingGap(BaseModel):
    """A window where a service did not submit, so absent days are not zeros.

    Nothing in the data separates a day with no incidents from a day that was
    never submitted. The distinction comes from the published guidance, so it
    is declared here and the panel build writes those days as null.
    """

    e_code_territory: str
    frs_territory: str
    start: date
    end: date
    source: str

    @model_validator(mode="after")
    def check_order(self) -> ReportingGap:
        if self.end < self.start:
            raise ValueError(
                f"reporting gap for {self.frs_territory} ends before it starts"
            )
        return self


class Validation(BaseModel):
    reconciliation_tolerance: float | None = None
    min_nonzero_days: int | None = Field(default=None, gt=0)
    reporting_gaps: list[ReportingGap] = []


class Weather(BaseModel):
    dataset: str
    catalogue_uuid: str | None = None
    version_directory: str | None = None
    resolution: str
    frequency: str
    variables: list[str] | None = None
    window_start: date
    release_covers_to: date
    base_url: str | None = None
    fill_value: float | None = None
    observing_window: str | None = None
    label_offset_days: dict[str, int] | None = None

    @model_validator(mode="after")
    def check_offsets(self) -> Weather:
        if self.label_offset_days is None or self.variables is None:
            return self
        missing = [
            name for name in self.variables if name not in self.label_offset_days
        ]
        if missing:
            raise ValueError(
                "weather.label_offset_days does not cover every configured "
                "variable.\n"
                f"  expected an offset for each of: {self.variables}\n"
                f"  found offsets for:              {sorted(self.label_offset_days)}\n"
                f"  missing:                        {missing}"
            )
        return self


BOUNDARY_COLLECTION = "ons_fra_boundaries"


class BoundaryMerge(BaseModel):
    """A boundary polygon folded into a recorded territory.

    The boundary layer carries the polygon separately while the incident data
    publishes it inside another territory. Neither source states the merge in
    a form that can be joined on, so it is declared here with its source.
    """

    boundary_code: str
    boundary_name: str
    e_code_territory: str
    frs_territory: str
    source: str


class Boundaries(BaseModel):
    layer: str
    merged: list[BoundaryMerge] = []


class Config(BaseModel):
    paths: Paths
    sources: dict[str, list[Source]]
    boundaries: Boundaries
    panel: Panel
    incident_types: IncidentTypes
    horizon: Horizon
    model: Model
    benchmark: Benchmark
    validation: Validation
    weather: Weather

    @field_validator("sources")
    @classmethod
    def check_sources(cls, value: dict[str, list[Source]]) -> dict[str, list[Source]]:
        for key, entries in value.items():
            names = [entry.filename for entry in entries]
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate filename in sources.{key}")
        return value

    @model_validator(mode="after")
    def check_boundary_layer(self) -> Config:
        available = [entry.era for entry in self.sources.get(BOUNDARY_COLLECTION, [])]
        if self.boundaries.layer not in available:
            raise ValueError(
                f"boundaries.layer does not name a layer under "
                f"sources.{BOUNDARY_COLLECTION}.\n"
                f"  expected one of: {available}\n"
                f"  found:           {self.boundaries.layer}"
            )
        return self


def load_config(path: Path | None = None) -> Config:
    path = path or repo_root() / "config" / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return Config.model_validate(raw)


INCIDENT_COLLECTION = "mhclg_daily_incidents"


def incident_sources(config: Config) -> list[Source]:
    """Return the incident file sources only, never the boundary or other layers."""
    return config.sources[INCIDENT_COLLECTION]


def boundary_layer(config: Config) -> Source:
    """Return the boundary source that boundaries.layer names.

    Loading the configuration has already checked that the name matches a
    layer under sources.ons_fra_boundaries.
    """
    return next(
        entry
        for entry in config.sources[BOUNDARY_COLLECTION]
        if entry.era == config.boundaries.layer
    )


def require(value: Any, name: str, note: str = "") -> Any:
    """Return a configured value, or fail naming the field that is undecided."""
    if value is None or value == [] or value == {}:
        message = (
            f"configuration field '{name}' is still undecided (null in config.yaml). "
            f"This stage cannot run until it is set."
        )
        if note:
            message = f"{message} {note}"
        raise UndecidedError(message)
    return value