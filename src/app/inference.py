from __future__ import annotations

from functools import lru_cache
from typing import Any
import warnings

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import anderson_ksamp, cramervonmises_2samp, ks_2samp, wasserstein_distance

from cgans import CGAN
from local_distributions import mftr
from local_datasets.mftr_uniform_conditional_dataset import normalize_conds_minus1_1

from .artifacts import VersionArtifacts
from .config import DEFAULT_BATCH_SIZE, SCENARIOS_DATA_ROOT


def _validate_params(m: float, mu: float, K: float, delta: float, omega: float) -> str | None:
    if m < 1.0:
        return "m must be >= 1"
    if mu < 1.0:
        return "mu must be >= 1"
    if K < 0.0:
        return "K must be >= 0"
    if not (0.0 <= delta < 1.0):
        return "delta must satisfy 0 <= delta < 1"
    if omega <= 0.0:
        return "omega must be > 0"
    return None


@lru_cache(maxsize=8)
def load_model_cached(checkpoint_path: str) -> CGAN:
    model = CGAN.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    return model


@lru_cache(maxsize=2)
def mftr_sampler(dist_type: str):
    return mftr(name="mftr", n_inverse_terms=2000, dist_type=dist_type)


def _common_hist_density(x: np.ndarray, y: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, int(bins) + 1)
    px, _ = np.histogram(x, bins=edges, density=True)
    py, _ = np.histogram(y, bins=edges, density=True)

    eps = 1e-12
    px = px.astype(float) + eps
    py = py.astype(float) + eps
    px /= px.sum()
    py /= py.sum()
    return px, py


def _metrics_two_sample(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = min(len(x), len(y))
    x = np.asarray(x[:n], dtype=float)
    y = np.asarray(y[:n], dtype=float)

    ks = ks_2samp(x, y)
    cvm = cramervonmises_2samp(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ad = anderson_ksamp([x, y])

    ks_stat = getattr(ks, "statistic", None)
    ks_pvalue = getattr(ks, "pvalue", None)
    if ks_stat is None or ks_pvalue is None:
        ks_stat, ks_pvalue = ks[0], ks[1]

    ks_stat_f = float(np.asarray(ks_stat).reshape(()).item())
    ks_pvalue_f = float(np.asarray(ks_pvalue).reshape(()).item())

    ad_stat = float(np.asarray(getattr(ad, "statistic", ad[0])).reshape(()).item())
    ad_pvalue = getattr(ad, "pvalue", None)
    if ad_pvalue is None:
        ad_sig = float(np.asarray(getattr(ad, "significance_level", 0.0)).reshape(()).item())
        ad_pvalue = ad_sig / 100.0

    px, py = _common_hist_density(x, y)
    js_divergence = float(jensenshannon(px, py) ** 2)
    hellinger = float(np.sqrt(0.5 * np.sum((np.sqrt(px) - np.sqrt(py)) ** 2)))

    return {
        "n": float(n),
        "ks_stat": ks_stat_f,
        "ks_pvalue": ks_pvalue_f,
        "cvm_stat": float(cvm.statistic),
        "cvm_pvalue": float(cvm.pvalue),
        "ad_stat": ad_stat,
        "ad_pvalue": float(np.asarray(ad_pvalue).reshape(()).item()),
        "js_divergence": js_divergence,
        "hellinger": hellinger,
        "wasserstein": float(wasserstein_distance(x, y)),
    }


def _resolve_normalization(
    art: VersionArtifacts,
) -> tuple[bool, tuple[tuple[float, float], ...]]:
    normalize_conds = bool(
        art.train_details.get("datasets", {}).get("train", {}).get(
            "normalize_conds",
            art.eval_metadata.get("normalize_conds", True),
        )
    )

    raw_ranges = art.scenario_source.get("cond_ranges", {})
    if art.is_scenario_run and raw_ranges:
        ranges = tuple(
            tuple(raw_ranges[name])
            for name in ["m", "mu", "K", "delta", "omega"]
        )
    else:
        ranges = tuple(art.ranges[name] for name in ["m", "mu", "K", "delta", "omega"])

    return normalize_conds, ranges


def run_comparison(
    *,
    art: VersionArtifacts,
    m: float,
    mu: float,
    K: float,
    delta: float,
    omega: float,
    n_samples: int,
    seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], str | None]:
    err = _validate_params(m, mu, K, delta, omega)
    if err is not None:
        return np.array([]), np.array([]), {}, err

    if art.checkpoint is None:
        return np.array([]), np.array([]), {}, "No checkpoint found for selected model version"

    model = load_model_cached(str(art.checkpoint))
    normalize_conds, ranges = _resolve_normalization(art)
    dist_type = str(art.eval_metadata.get("dist_type", "amplitude"))

    cond_raw = torch.tensor([[m, mu, K, delta, omega]], dtype=torch.float32)
    if normalize_conds:
        conds = normalize_conds_minus1_1(cond_raw, ranges)
    else:
        conds = cond_raw

    conds_rep = conds.repeat(int(n_samples), 1)

    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(int(seed))

    outputs = []
    with torch.inference_mode():
        for start in range(0, int(n_samples), int(batch_size)):
            c = conds_rep[start : start + int(batch_size)].to(device)
            z = torch.randn(c.shape[0], int(model.latent_dim), device=device, generator=gen)
            x = model(z, c)
            outputs.append(x.detach().cpu())
    generated = torch.cat(outputs, dim=0).reshape(-1).numpy().astype(float)

    sampler = mftr_sampler(dist_type)
    real = sampler.rvs(
        float(m),
        float(K),
        float(delta),
        float(mu),
        float(omega),
        size=int(n_samples),
        random_state=int(seed),
    )
    real = np.asarray(real, dtype=float).reshape(-1)

    metrics = _metrics_two_sample(real, generated)
    return real, generated, metrics, None


def run_scenario_comparison(
    *,
    art: VersionArtifacts,
    m: float,
    mu: float,
    K: float,
    delta: float,
    omega: float,
    scenario_id: int,
    n_samples: int,
    seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], str | None]:
    """Compare scenario empirical samples against model-generated samples.

    Unlike `run_comparison`, the 'real' samples come from the scenario's
    `samples.csv` file rather than from the MFTR theoretical distribution.
    Parameter validation against strict MFTR domain constraints is intentionally
    skipped here, because scenario-estimated params may lie outside those bounds
    (e.g. mu < 1, delta == 1).
    """
    if art.checkpoint is None:
        return np.array([]), np.array([]), {}, "No checkpoint found for selected model version"

    model = load_model_cached(str(art.checkpoint))

    normalize_conds, ranges = _resolve_normalization(art)

    cond_raw = torch.tensor([[m, mu, K, delta, omega]], dtype=torch.float32)
    if normalize_conds:
        conds = normalize_conds_minus1_1(cond_raw, ranges)
    else:
        conds = cond_raw

    conds_rep = conds.repeat(int(n_samples), 1)
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(int(seed))

    outputs = []
    with torch.inference_mode():
        for start in range(0, int(n_samples), int(batch_size)):
            c = conds_rep[start : start + int(batch_size)].to(device)
            z = torch.randn(c.shape[0], int(model.latent_dim), device=device, generator=gen)
            x = model(z, c)
            outputs.append(x.detach().cpu())
    generated = torch.cat(outputs, dim=0).reshape(-1).numpy().astype(float)

    from local_datasets.scenario_data import load_scenario_data
    scenario = load_scenario_data(scenario_id, SCENARIOS_DATA_ROOT)
    real = scenario.load_samples().astype(float)

    metrics = _metrics_two_sample(real, generated)
    return real, generated, metrics, None


def metrics_table(metrics: dict[str, float]) -> list[list[Any]]:
    if not metrics:
        return []

    def r5(x: float) -> str:
        return f"{x:.5e}"

    return [
        ["KS statistic", r5(metrics["ks_stat"]), "Maximum ECDF distance"],
        ["KS p-value", r5(metrics["ks_pvalue"]), "Significance for KS test"],
        ["CvM statistic", r5(metrics["cvm_stat"]), "Integrated ECDF mismatch"],
        ["CvM p-value", r5(metrics["cvm_pvalue"]), "Significance for CvM test"],
        ["AD statistic", r5(metrics["ad_stat"]), "Tail-sensitive two-sample discrepancy"],
        ["AD p-value", r5(metrics["ad_pvalue"]), "Significance for AD two-sample test"],
        ["JS divergence", r5(metrics["js_divergence"]), "Symmetric divergence between estimated densities"],
        ["Hellinger distance", r5(metrics["hellinger"]), "Distance between estimated density distributions"],
        ["Wasserstein", r5(metrics["wasserstein"]), "Earth mover distance (1D)"],
    ]
