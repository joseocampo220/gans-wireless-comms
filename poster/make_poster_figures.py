#!/usr/bin/env python3
"""
make_poster_figures.py — Generates all figures for the MFTR cGAN poster.

Usage (from repo root):
    uv run python poster/make_poster_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd

from cgans import CGAN
from local_distributions import mftr as mftr_dist_factory
from local_distributions.mftr_dist import pdf_mftr
from local_datasets.mftr_uniform_conditional_dataset import normalize_conds_minus1_1

# Paths
_FIGURES_DIR   = _REPO_ROOT / "poster" / "figures"
_CKPT          = (_REPO_ROOT / "logs" / "mftr" / "cgan" / "version_19"
                  / "checkpoints" / "epoch=19-step=325540.ckpt")
_SCENARIOS_DIR = _REPO_ROOT / "data" / "scenarios"

_RANGES = {"m": (1.0,8.0), "mu": (1.0,9.0), "K": (0.0,20.0),
           "delta": (0.0,0.9), "omega": (1.0,2.0)}

_RC = {
    "font.family": "serif", "font.serif": ["Times New Roman","Times","DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "text.usetex": False,
    "font.size": 14, "axes.labelsize": 14, "axes.titlesize": 14,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "lines.linewidth": 2.2, "axes.grid": True, "grid.alpha": 0.22,
    "grid.linewidth": 0.55, "axes.axisbelow": True,
    "axes.edgecolor": "#1f2937", "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": True, "legend.fancybox": False, "legend.framealpha": 1.0,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.dpi": 150,
}

C_RED    = "#ED1C24"
C_BLUE   = "#0b84a5"
C_ORANGE = "#d97706"
C_HIST   = "#9aa5b1"
C_GREY   = "#6b7280"


def _apply_style():
    plt.rcParams.update(_RC)


def _mftr_amp_pdf(r, *, m, mu, K, delta, omega, n_terms=3000):
    r = np.asarray(r, dtype=float)
    f_g = pdf_mftr(r**2, m=float(m), K=float(K), delta=float(delta),
                   mu=float(mu), omega=float(omega), n_inverse_terms=n_terms)
    return f_g * (2.0 * r)


def _strip(ax):
    """Remove all tick labels and axis labels — maximise data-ink ratio."""
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel(""); ax.set_ylabel("")


def _ecdf(a):
    xs = np.sort(a)
    ys = np.arange(1, xs.size + 1, dtype=float) / xs.size
    return xs, ys


# ---------------------------------------------------------------------------
# Figure 1: MFTR shape
# ---------------------------------------------------------------------------

def make_mftr_shape():
    _apply_style()
    case_a = dict(delta=0.9, m=8.0,  K=8.0,  omega=2.0)
    case_b = dict(delta=0.9, m=4.0,  K=15.0, omega=1.0)
    styles = {1: (C_ORANGE,":"), 3: (C_BLUE,"--"), 5: (C_RED,"-."), 7: ("black","-")}
    r = np.linspace(1e-5, 2.5, 2000)

    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    for mu, (col, ls) in styles.items():
        y_a = _mftr_amp_pdf(r, mu=float(mu), **case_a)
        y_b = _mftr_amp_pdf(r, mu=float(mu), **case_b)
        ax.plot(r, y_a, color=col, linestyle=ls, label=rf"$\mu={mu}$")
        ax.plot(r, y_b, color=col, linestyle=ls)
    ax.annotate("B", xy=(0.52,0.90), fontsize=9, color=C_GREY)
    ax.annotate("A", xy=(1.25,0.48), fontsize=9, color=C_GREY)
    ax.set_xlim(0,2.5); ax.set_ylim(0,1.35)
    ax.set_xlabel(r"$r$", labelpad=1); ax.set_ylabel(r"$f_R(r)$", labelpad=1)
    ax.legend(loc="upper right", ncol=2, fontsize=9, handlelength=1.4,
              borderpad=0.25, labelspacing=0.25, borderaxespad=0.3)
    fig.tight_layout(pad=0.15)
    out = _FIGURES_DIR / "mftr_shape.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# Figure 2: Scenario 1 — empirical PDF vs. MFTR fits
# ---------------------------------------------------------------------------

def make_scenario1_figure():
    _apply_style()
    csv = _SCENARIOS_DIR / "scenario_1" / "pdf_data.csv"
    df  = pd.read_csv(csv)
    x, emp = df["xi"].values, df["pdf_empirical"].values
    fms, ga = df["pdf_fminsearch"].values, df["pdf_ga"].values

    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    ax.plot(x, emp, color=C_GREY,   ls="-",  lw=1.8, label="Empirical")
    ax.plot(x, fms, color=C_RED,    ls="--", lw=2.0, label="fminsearch")
    ax.plot(x, ga,  color=C_BLUE,   ls=":",  lw=2.0, label="GA")
    _strip(ax)
    ax.legend(fontsize=10, loc="upper right", handlelength=1.5,
              borderpad=0.25, labelspacing=0.25)
    fig.tight_layout(pad=0.15)
    out = _FIGURES_DIR / "scenario1.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# Figures 3–6: Result figures — fully stripped, no labels/ticks/legends
# ---------------------------------------------------------------------------

_MODEL_CACHE = None


def _load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        print(f"  Loading {_CKPT.name} …")
        _MODEL_CACHE = CGAN.load_from_checkpoint(str(_CKPT), map_location="cpu")
        _MODEL_CACHE.eval()
        print("  OK.")
    return _MODEL_CACHE


def _generate(model, *, m, mu, K, delta, omega, n=8000, seed=42):
    cond_raw = torch.tensor([[m, mu, K, delta, omega]], dtype=torch.float32)
    conds = normalize_conds_minus1_1(
        cond_raw,
        (_RANGES["m"],_RANGES["mu"],_RANGES["K"],_RANGES["delta"],_RANGES["omega"]),
    ).repeat(n, 1)
    gen = torch.Generator().manual_seed(seed)
    out = []
    with torch.inference_mode():
        for i in range(0, n, 2048):
            c = conds[i:i+2048]
            z = torch.randn(c.shape[0], model.latent_dim, generator=gen)
            out.append(model(z, c).detach())
    return torch.cat(out).reshape(-1).numpy().astype(float)


def _reference(*, m, mu, K, delta, omega, n=8000, seed=42):
    s = mftr_dist_factory(name="mftr", n_inverse_terms=2000, dist_type="amplitude")
    return np.asarray(s.rvs(m,K,delta,mu,omega,size=n,random_state=seed),dtype=float).reshape(-1)


def make_result_figure(*, m, mu, K, delta, omega, filename, n=8000, seed=42):
    """Fully stripped: no axis labels, no tick labels, no legend."""
    _apply_style()
    model    = _load_model()
    gen_samp = _generate(model, m=m,mu=mu,K=K,delta=delta,omega=omega,n=n,seed=seed)
    ref_samp = _reference(m=m,mu=mu,K=K,delta=delta,omega=omega,n=n,seed=seed)

    fig, (ax_d, ax_c) = plt.subplots(1, 2, figsize=(5.6, 2.0),
                                      gridspec_kw={"wspace": 0.06})

    # Density (histogram + KDE + MFTR PDF)
    sns.histplot(gen_samp, ax=ax_d, stat="density", color=C_HIST, alpha=0.22,
                 bins=55, edgecolor="none")
    sns.kdeplot(gen_samp, ax=ax_d, color=C_BLUE, linewidth=2.2)
    r_hi = max(ref_samp.max(), gen_samp.max()) * 1.05
    r_g  = np.linspace(1e-5, r_hi, 1000)
    ax_d.plot(r_g, _mftr_amp_pdf(r_g,m=m,mu=mu,K=K,delta=delta,omega=omega),
              color=C_RED, linewidth=2.2)
    _strip(ax_d)
    ax_d.margins(x=0.02, y=0.05)

    # ECDF
    xs_r, ys_r = _ecdf(ref_samp)
    xs_g, ys_g = _ecdf(gen_samp)
    ax_c.plot(xs_r, ys_r, color=C_RED,  lw=2.0)
    ax_c.plot(xs_g, ys_g, color=C_BLUE, lw=2.0, ls="--")
    _strip(ax_c)
    ax_c.margins(x=0.02, y=0.02)

    fig.tight_layout(pad=0.1)
    out = _FIGURES_DIR / filename
    fig.savefig(out); plt.close(fig)
    print(f"  [OK] {out}")


RESULT_CASES = [
    dict(m=5.0,  mu=2.0,  K=8.0,  delta=0.70, omega=1.5, filename="result_1.pdf"),
    dict(m=6.0,  mu=7.0,  K=12.0, delta=0.85, omega=1.3, filename="result_2.pdf"),
    dict(m=5.0,  mu=12.0, K=8.0,  delta=0.70, omega=1.5, filename="result_3.pdf"),
    dict(m=10.0, mu=5.0,  K=8.0,  delta=0.70, omega=1.5, filename="result_4.pdf"),
]


def main():
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating figures → {_FIGURES_DIR}\n")
    print("1  MFTR shape …");        make_mftr_shape()
    print("2  Scenario 1 …");        make_scenario1_figure()
    for i, c in enumerate(RESULT_CASES, 3):
        tag = "Interp" if i < 5 else "Extrap"
        print(f"{i}  Result ({tag}) — {c['filename']} …")
        make_result_figure(**c)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
