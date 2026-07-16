"""Minimal tests for standardized scenario data loading.

These tests intentionally cover only fundamentals required by the new
scenario-ingestion baseline:
1. Scenario discovery under `data/scenarios`.
2. Successful parse of one scenario with expected condition-vector shape/order.
3. Required-file validation failure on incomplete scenario folders.
"""

from pathlib import Path

import pytest

from local_datasets import list_available_scenarios, load_scenario_data


def test_list_available_scenarios_in_repo_data():
    root = Path(__file__).resolve().parents[1] / "data" / "scenarios"
    scenario_ids = list_available_scenarios(root)
    assert scenario_ids == [1, 2, 3, 4, 5, 6]


def test_load_scenario_data_fundamentals():
    root = Path(__file__).resolve().parents[1] / "data" / "scenarios"
    scenario = load_scenario_data(1, root, method="fminsearch")

    assert scenario.scenario_id == 1
    assert scenario.selected_method == "fminsearch"
    assert scenario.selected_condition.shape == (5,)
    assert scenario.samples is None
    assert scenario.samples_path.name == "samples.csv"
    assert scenario.xi.ndim == 1
    assert scenario.xi.size > 0

    loaded_samples = scenario.load_samples()
    assert loaded_samples.ndim == 1
    assert loaded_samples.size > 0


def test_load_scenario_data_missing_required_files(tmp_path: Path):
    scenarios_root = tmp_path / "scenarios"
    scenario_dir = scenarios_root / "scenario_1"
    scenario_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        load_scenario_data(1, scenarios_root)
