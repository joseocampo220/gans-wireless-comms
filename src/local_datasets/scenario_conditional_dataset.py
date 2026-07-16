"""Scenario-conditioned dataset for scenario-only CGAN training.

This module provides a single dataset class that supports both scenario data
sources needed for phase-1 scenario-only experiments:
1. `empirical_pdf`: sample values from empirical PDF XY coordinates.
2. `mftr_generated`: generate values from MFTR using scenario estimated params.

Organization
1. A continuous sampler for empirical PDF coordinates.
2. `ScenarioConditionalDataset` that loads selected scenarios and builds a
   `(sample, condition)` dataset compatible with the existing CGAN pipeline.

Design notes
- Conditions always follow model order: `[m, mu, K, delta, omega]`.
- One condition row is associated with each selected scenario.
- `samples_per_scenario` controls how many samples are generated per scenario.
- Output tensors follow the same contract as existing conditional datasets:
  `sample.shape == (1,)` and `condition.shape == (5,)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import rv_histogram
import torch
from torch.utils.data import Dataset

from local_distributions import mftr
from local_datasets.scenario_data import load_scenario_data


def sample_from_empirical_pdf(
    xi: np.ndarray,
    pdf_values: np.ndarray,
    num_samples: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw samples from an empirical PDF represented by `(xi, pdf_values)`.

    Sampling uses a SciPy `rv_histogram` distribution constructed from the
    tabulated empirical PDF. This yields continuous samples inside each bin,
    which is typically a better fit for downstream density matching than
    discrete grid-point sampling.
    """
    x = np.asarray(xi, dtype=np.float64).reshape(-1)
    p = np.asarray(pdf_values, dtype=np.float64).reshape(-1)

    if x.size < 2 or p.size != x.size:
        raise ValueError("xi and pdf_values must be 1D arrays with equal length >= 2")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    order = np.argsort(x)
    x = x[order]
    p = p[order]

    p = np.maximum(p, 0.0)
    if not np.any(p > 0.0):
        raise ValueError("Empirical PDF must contain at least one positive value")

    if np.any(np.diff(x) <= 0):
        raise ValueError("xi values must be strictly increasing after sorting")

    # Convert support points into bin edges using midpoint boundaries.
    # First/last bins are extended symmetrically to preserve edge spacing.
    mid = 0.5 * (x[:-1] + x[1:])
    left = x[0] - 0.5 * (x[1] - x[0])
    right = x[-1] + 0.5 * (x[-1] - x[-2])
    edges = np.concatenate(([left], mid, [right]))

    dist = rv_histogram((p, edges), density=True)
    samples = dist.rvs(size=int(num_samples), random_state=rng)
    return np.asarray(samples, dtype=np.float32)


class ScenarioConditionalDataset(Dataset):
    """Conditional dataset built from selected scenarios.

    Parameters
    - `scenario_ids`: Scenario identifiers to include in the dataset.
    - `scenarios_root`: Path to the directory containing `scenario_<id>` folders.
    - `samples_per_scenario`: Number of generated samples per scenario.
    - `source`: `empirical_pdf` or `mftr_generated`.
    - `method`: Estimated-params method from `estimated_params.csv`.
    - `dist_type`: Distribution output for MFTR generation (`amplitude` or `power`).
    - `seed`: Optional seed used for reproducible sampling.
    """

    def __init__(
        self,
        *,
        scenario_ids: list[int],
        scenarios_root: Path | str,
        samples_per_scenario: int,
        source: str = "empirical_pdf",
        method: str = "fminsearch",
        dist_type: str = "amplitude",
        seed: Optional[int] = None,
    ):
        if not scenario_ids:
            raise ValueError("scenario_ids must not be empty")

        self.scenario_ids = [int(sid) for sid in scenario_ids]
        self.scenarios_root = Path(scenarios_root)
        self.samples_per_scenario = int(samples_per_scenario)
        if self.samples_per_scenario <= 0:
            raise ValueError("samples_per_scenario must be > 0")

        allowed_sources = {"empirical_pdf", "mftr_generated"}
        if source not in allowed_sources:
            raise ValueError(f"source must be one of {sorted(allowed_sources)}")
        self.source = source

        if dist_type not in ("amplitude", "power"):
            raise ValueError("dist_type must be 'amplitude' or 'power'")
        self.dist_type = dist_type

        self.method = method
        self.seed = seed

        self.scenarios = [
            load_scenario_data(sid, self.scenarios_root, method=self.method)
            for sid in self.scenario_ids
        ]

        self.conds_raw = torch.tensor(
            [s.selected_condition.tolist() for s in self.scenarios],
            dtype=torch.float32,
        )

        rng = np.random.default_rng(seed)
        mftr_gen = (
            mftr(name="mftr", n_inverse_terms=2000, dist_type=self.dist_type)
            if self.source == "mftr_generated"
            else None
        )

        samples_all: list[torch.Tensor] = []
        for i, scenario in enumerate(self.scenarios):
            if self.source == "empirical_pdf":
                values = sample_from_empirical_pdf(
                    scenario.xi,
                    scenario.pdf_empirical,
                    self.samples_per_scenario,
                    rng=rng,
                )
            else:
                m_i, mu_i, K_i, delta_i, omega_i = (
                    float(x) for x in scenario.selected_condition
                )
                values = mftr_gen.rvs(
                    m_i,
                    K_i,
                    delta_i,
                    mu_i,
                    omega_i,
                    size=self.samples_per_scenario,
                    random_state=None if seed is None else int(seed) + i,
                )
                values = np.asarray(values, dtype=np.float32)

            samples_all.append(
                torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
            )

        self.samples = torch.cat(samples_all, dim=0)

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: int):
        scenario_idx = idx // self.samples_per_scenario
        return self.samples[idx], self.conds_raw[scenario_idx]
