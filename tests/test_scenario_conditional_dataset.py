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


def test_scenario_conditional_dataset_cond_ranges_override():
    """When cond_ranges is provided, normalization uses those ranges
    rather than computing ranges from the data."""
    dataset = ScenarioConditionalDataset(
        scenario_ids=[1, 2],
        scenarios_root=_scenarios_root(),
        samples_per_scenario=16,
        source="empirical_pdf",
        method="fminsearch",
        normalize_conds=True,
        cond_ranges=(
            (0.0, 100.0),  # m
            (0.0, 100.0),  # mu
            (0.0, 100.0),  # K
            (0.0, 1.0),  # delta
            (0.0, 10.0),  # omega
        ),
        seed=42,
    )

    assert dataset.cond_ranges == (
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 1.0),
        (0.0, 10.0),
    )

    # Verify the passed ranges were actually used: normalise scenario-2 params
    # back from normalised to raw and compare to the actual conds_raw.
    # This proves the custom ranges, not the data-derived ones, were applied.
    raw = dataset.conds_raw[1]  # scenario 2 params
    nrm = dataset.conds[1]  # normalised using custom ranges
    for i, (lo, hi) in enumerate(
        [(0.0, 100.0), (0.0, 100.0), (0.0, 100.0), (0.0, 1.0), (0.0, 10.0)]
    ):
        expected = 2.0 * (float(raw[i]) - lo) / (hi - lo) - 1.0
        assert abs(float(nrm[i]) - expected) < 1e-5, (
            f"Param {i}: raw={float(raw[i]):.4f}, expected={expected:.4f}, "
            f"got={float(nrm[i]):.4f}"
        )


def test_scenario_conditional_dataset_cond_ranges_override_no_normalize():
    """cond_ranges is stored but ignored when normalize_conds=False."""
    dataset = ScenarioConditionalDataset(
        scenario_ids=[1],
        scenarios_root=_scenarios_root(),
        samples_per_scenario=8,
        source="empirical_pdf",
        method="fminsearch",
        normalize_conds=False,
        cond_ranges=(
            (0.0, 100.0),
            (0.0, 100.0),
            (0.0, 100.0),
            (0.0, 1.0),
            (0.0, 10.0),
        ),
        seed=42,
    )

    # Explicit ranges are stored even when not normalising.
    assert dataset.cond_ranges == (
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 1.0),
        (0.0, 10.0),
    )

    # conds equals conds_raw (no transform).
    assert dataset.conds.equal(dataset.conds_raw)
