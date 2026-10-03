"""Tests for configuration loading.

Cover the schema accepting config.yaml as written, the horizon grid being
whole, every exclusion carrying a source, and `require` raising on a null
value rather than returning a default.
"""

from __future__ import annotations

import pytest

from fdh.config import UndecidedError, load_config, require


def test_config_loads() -> None:
    config = load_config()
    assert config.horizon.lead_times[0] == config.horizon.min_days
    assert config.model.information_condition in ("restricted", "perfect_foresight")


def test_undecided_panel_raises_naming_the_field() -> None:
    config = load_config()
    if config.panel.services is None:
        with pytest.raises(UndecidedError, match="panel.services"):
            require(config.panel.services, "panel.services")


def test_horizon_grid_is_whole() -> None:
    config = load_config()
    lead_times = config.horizon.lead_times
    assert lead_times[-1] == config.horizon.max_days
    assert all(
        second - first == config.horizon.step_days
        for first, second in zip(lead_times, lead_times[1:], strict=False)
    )


def test_every_exclusion_cites_a_source() -> None:
    config = load_config()
    for exclusion in config.exclusions:
        assert exclusion.source.strip(), f"{exclusion.service} exclusion has no source"


def test_source_hashes_are_lowercase_hex_when_present() -> None:
    config = load_config()
    for sources in config.sources.values():
        for source in sources:
            if source.sha256:
                assert len(source.sha256) == 64
                assert source.sha256 == source.sha256.lower()