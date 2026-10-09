from pathlib import Path

from hea_forecast.config import load_config


def test_project_config_has_non_overlapping_cutoff_and_horizon() -> None:
    config = load_config(Path("configs/project.yaml"))
    for window in config["temporal_backtests"]:
        assert window["cutoff_year"] < window["horizon_start"]
        assert window["horizon_start"] <= window["horizon_end"]


def test_core_reactions_are_configured() -> None:
    config = load_config(Path("configs/project.yaml"))
    assert set(config["reactions"]) == {"HER", "OER", "ORR", "CO2RR"}
