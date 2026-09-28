"""Tests for inference conditioning routing.

Verifies that both run_comparison and run_scenario_comparison correctly
condition generated samples on the selected parameters, regardless of
whether the model is a scenario run or not.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from app.artifacts import VersionArtifacts
from app.inference import _resolve_normalization, run_comparison, run_scenario_comparison
from cgans import CGAN


def _make_artifacts(
    *,
    is_scenario=False,
    cond_ranges=None,
    eval_ranges=None,
    checkpoint=None,
    normalize_conds=True,
):
    art = MagicMock(spec=VersionArtifacts)
    art.checkpoint = checkpoint
    art.is_scenario_run = is_scenario

    if is_scenario:
        art.scenario_source = {"cond_ranges": cond_ranges or {}}
    else:
        art.scenario_source = {}

    art.ranges = {
        "m": (1.0, 100.0),
        "mu": (1.0, 100.0),
        "K": (0.0, 50.0),
        "delta": (0.0, 1.0),
        "omega": (0.1, 10.0),
    }
    if eval_ranges:
        art.ranges = eval_ranges

    art.eval_metadata = {"normalize_conds": normalize_conds}
    art.train_details = {
        "datasets": {"train": {"normalize_conds": normalize_conds}},
        "run_type": "train_cgan_scenarios" if is_scenario else "train_cgan",
    }
    return art


class TestResolveNormalization:
    def test_scenario_run_uses_cond_ranges(self):
        cond_ranges = {
            "m": [1.0, 500.0],
            "mu": [0.5, 10.0],
            "K": [0.0, 100.0],
            "delta": [0.0, 1.0],
            "omega": [0.1, 5.0],
        }
        art = _make_artifacts(is_scenario=True, cond_ranges=cond_ranges)
        normalize_conds, ranges = _resolve_normalization(art)

        assert normalize_conds is True
        assert ranges[0] == (1.0, 500.0)
        assert ranges[1] == (0.5, 10.0)
        assert ranges[2] == (0.0, 100.0)
        assert ranges[3] == (0.0, 1.0)
        assert ranges[4] == (0.1, 5.0)

    def test_non_scenario_run_uses_art_ranges(self):
        art = _make_artifacts(is_scenario=False)
        normalize_conds, ranges = _resolve_normalization(art)

        assert normalize_conds is True
        assert ranges[0] == (1.0, 100.0)
        assert ranges[1] == (1.0, 100.0)

    def test_scenario_run_without_cond_ranges_falls_back(self):
        art = _make_artifacts(is_scenario=True, cond_ranges={})
        normalize_conds, ranges = _resolve_normalization(art)

        assert ranges[0] == (1.0, 100.0)

    def test_normalize_conds_false(self):
        art = _make_artifacts(is_scenario=True, normalize_conds=False)
        normalize_conds, _ = _resolve_normalization(art)

        assert normalize_conds is False

    def test_normalize_conds_from_train_details(self):
        art = _make_artifacts(is_scenario=False, normalize_conds=False)
        art.train_details = {
            "datasets": {"train": {"normalize_conds": True}},
            "run_type": "train_cgan",
        }
        normalize_conds, _ = _resolve_normalization(art)

        assert normalize_conds is True


class TestConditioningEndToEnd:
    @pytest.fixture
    def cgan_model(self):
        return CGAN(latent_dim=10, cond_dim=5, cond_emb_dim=8)

    @pytest.fixture
    def scenario_art(self, cgan_model, monkeypatch):
        monkeypatch.setattr(
            "app.inference.load_model_cached", lambda _: cgan_model
        )
        cond_ranges = {
            "m": [1.0, 100.0],
            "mu": [1.0, 100.0],
            "K": [0.0, 50.0],
            "delta": [0.0, 1.0],
            "omega": [0.1, 10.0],
        }
        return _make_artifacts(
            is_scenario=True,
            cond_ranges=cond_ranges,
            checkpoint=Path("mock.ckpt"),
        )

    def test_run_comparison_scenario_model_different_params(self, scenario_art):
        _, gen1, _, err1 = run_comparison(
            art=scenario_art,
            m=5.0, mu=5.0, K=5.0, delta=0.5, omega=1.0,
            n_samples=500, seed=42,
        )
        _, gen2, _, err2 = run_comparison(
            art=scenario_art,
            m=50.0, mu=50.0, K=50.0, delta=0.9, omega=5.0,
            n_samples=500, seed=42,
        )

        assert err1 is None
        assert err2 is None
        assert not np.allclose(gen1, gen2)

    def test_run_comparison_scenario_model_same_params_same_seed(self, scenario_art):
        _, gen1, _, err1 = run_comparison(
            art=scenario_art,
            m=5.0, mu=5.0, K=5.0, delta=0.5, omega=1.0,
            n_samples=500, seed=42,
        )
        _, gen2, _, err2 = run_comparison(
            art=scenario_art,
            m=5.0, mu=5.0, K=5.0, delta=0.5, omega=1.0,
            n_samples=500, seed=42,
        )

        assert err1 is None
        assert err2 is None
        assert np.allclose(gen1, gen2)

    def test_run_scenario_comparison_different_params(self, scenario_art):
        _, gen1, _, err1 = run_scenario_comparison(
            art=scenario_art,
            m=5.0, mu=5.0, K=5.0, delta=0.5, omega=1.0,
            scenario_id=1, n_samples=500, seed=42,
        )
        _, gen2, _, err2 = run_scenario_comparison(
            art=scenario_art,
            m=50.0, mu=50.0, K=50.0, delta=0.9, omega=5.0,
            scenario_id=1, n_samples=500, seed=42,
        )

        assert err1 is None
        assert err2 is None
        assert not np.allclose(gen1, gen2)

    def test_run_comparison_vs_scenario_different_real(self, scenario_art):
        real1, gen1, _, err1 = run_comparison(
            art=scenario_art,
            m=5.0, mu=5.0, K=5.0, delta=0.5, omega=1.0,
            n_samples=500, seed=42,
        )
        real2, gen2, _, err2 = run_scenario_comparison(
            art=scenario_art,
            m=5.0, mu=5.0, K=5.0, delta=0.5, omega=1.0,
            scenario_id=1, n_samples=500, seed=42,
        )

        assert err1 is None
        assert err2 is None
        assert real1.shape != real2.shape or not np.allclose(real1, real2)
        assert np.allclose(gen1, gen2)
