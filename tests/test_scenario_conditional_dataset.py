"""Minimal tests for ScenarioConditionalDataset fundamentals.

Scope intentionally stays minimal:
1. Dataset shape/contract for `empirical_pdf` source.
2. Dataset shape/contract for `mftr_generated` source.
3. Condition vector shape and scenario grouping behavior.
"""

from pathlib import Path

from local_datasets import ScenarioConditionalDataset


def _scenarios_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "scenarios"


def test_scenario_conditional_dataset_empirical_pdf_fundamentals():
    dataset = ScenarioConditionalDataset(
        scenario_ids=[1, 2],
        scenarios_root=_scenarios_root(),
        samples_per_scenario=32,
        source="empirical_pdf",
        method="fminsearch",
        seed=42,
    )

    assert len(dataset) == 64
    assert dataset.samples.shape == (64, 1)
    assert dataset.conds_raw.shape == (2, 5)

    x0, c0 = dataset[0]
    x1, c1 = dataset[31]
    x2, c2 = dataset[32]

    assert x0.shape == (1,)
    assert c0.shape == (5,)
    assert c1.shape == (5,)
    assert c2.shape == (5,)

    # First chunk maps to first scenario condition, second chunk to second scenario.
    assert c0.equal(c1)
    assert not c0.equal(c2)


def test_scenario_conditional_dataset_mftr_generated_fundamentals():
    dataset = ScenarioConditionalDataset(
        scenario_ids=[1],
        scenarios_root=_scenarios_root(),
        samples_per_scenario=16,
        source="mftr_generated",
        method="fminsearch",
        seed=7,
    )

    assert len(dataset) == 16
    assert dataset.samples.shape == (16, 1)
    assert dataset.conds_raw.shape == (1, 5)

    x0, c0 = dataset[0]
    x_last, c_last = dataset[15]

    assert x0.shape == (1,)
    assert c0.shape == (5,)
    assert x_last.shape == (1,)
    assert c_last.shape == (5,)
    assert c0.equal(c_last)
