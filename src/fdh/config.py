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


class Exclusion(BaseModel):
    service: str
    start: date
    end: date
    status: Literal["proposed", "confirmed"]
    source: str

    @model_validator(mode="after")
    def check_order(self) -> Exclusion:
        if self.end < self.start:
            raise ValueError(f"exclusion for {self.service} ends before it starts")
        return self


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


class Validation(BaseModel):
    reconciliation_tolerance: float | None = None


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
    ordering_tolerance: float | None = Field(default=None, ge=0, le=1)


class Config(BaseModel):
    paths: Paths
    sources: dict[str, list[Source]]
    panel: Panel
    incident_types: IncidentTypes
    exclusions: list[Exclusion] = []
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


def load_config(path: Path | None = None) -> Config:
    path = path or repo_root() / "config" / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"configuration not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return Config.model_validate(raw)


INCIDENT_COLLECTION = "mhclg_daily_incidents"


def incident_sources(config: Config) -> list[Source]:
    """Return the incident file sources only, never the boundary or other layers."""
    if INCIDENT_COLLECTION not in config.sources:
        raise KeyError(
            f"config.yaml sources has no '{INCIDENT_COLLECTION}' collection. "
            f"Found: {sorted(config.sources)}"
        )
    return config.sources[INCIDENT_COLLECTION]


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