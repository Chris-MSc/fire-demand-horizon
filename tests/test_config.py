from __future__ import annotations

import pytest
from pydantic import ValidationError

from fdh.config import Horizon, Panel, UndecidedError, load_config, require


def test_config_loads() -> None:
    config = load_config()
    assert config.horizon.lead_times[0] == config.horizon.min_days


@pytest.mark.parametrize("undecided", [None, [], {}])
def test_require_names_an_undecided_field(undecided: object) -> None:
    with pytest.raises(UndecidedError, match="panel.services"):
        require(undecided, "panel.services")


def test_require_returns_a_decided_value() -> None:
    assert require(["Suffolk"], "panel.services") == ["Suffolk"]


def test_horizon_rejects_a_grid_that_does_not_divide() -> None:
    with pytest.raises(ValidationError, match="not whole"):
        Horizon(min_days=7, max_days=80, step_days=7, mase_threshold=1.0)


def test_source_hashes_are_lowercase_hex_when_present() -> None:
    config = load_config()
    for sources in config.sources.values():
        for source in sources:
            if source.sha256:
                assert len(source.sha256) == 64
                assert source.sha256 == source.sha256.lower()


def test_rural_urban_classes_accept_published_names_and_null() -> None:
    assert Panel(rural_urban_classes=None).rural_urban_classes is None
    chosen = ["Majority rural", "Intermediate rural"]
    assert Panel(rural_urban_classes=chosen).rural_urban_classes == chosen


def test_rural_urban_classes_name_a_misspelt_class() -> None:
    with pytest.raises(ValidationError, match="Majority Rural"):
        Panel(rural_urban_classes=["Majority Rural", "Intermediate rural"])


def test_rural_urban_classes_reject_a_repeat() -> None:
    with pytest.raises(ValidationError, match="repeats"):
        Panel(rural_urban_classes=["Urban", "Urban"])
