from __future__ import annotations

import pytest
from pydantic import ValidationError

from fdh.config import Panel, UndecidedError, load_config, require


def test_the_project_config_loads() -> None:
    config = load_config()
    assert config.horizon.lead_times[0] == config.horizon.min_days


@pytest.mark.parametrize("undecided", [None, [], {}])
def test_require_stops_on_an_undecided_field(undecided: object) -> None:
    with pytest.raises(UndecidedError):
        require(undecided, "panel.services")


def test_a_misspelt_settlement_class_is_refused() -> None:
    with pytest.raises(ValidationError):
        Panel(rural_urban_classes=["Majority Rural", "Intermediate rural"])
