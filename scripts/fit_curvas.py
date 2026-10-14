"""Fit tabulated envelope PDFs to MFTR with integer mu; no empirical sampling.

Scope and assumptions
---------------------
CSV columns xi,pdf_empirical describe linear envelope amplitude. The PDF is
piecewise linear between unique points; missing mass outside its interval is
assumed zero. Normalize its area and second moment to Omega=E[R**2]=1.
Default search:
    - real K in [0,50]
    - Delta in [0,1]
    - real m in [0.1,60]
    - integer mu in {1,2,3,4,5}.

Earlier fits reached K~87 and m~500: results here
apply only to this bounded subfamily per request, not all MFTR. The theoretical model
admits positive real mu; this experiment intentionally tests integers only.

The simulator implements arXiv:2212.02448v2 equations (5),(7),(8): shared
zeta~Gamma(m,1/m) multiplies specular POWER, hence sqrt(zeta) multiplies
amplitudes. Assign all specular power to the two rays of cluster 1; U_i=0
for i>=2. This preserves the envelope law: conditional on zeta and relative
phase theta, power/sigma**2 is noncentral chi-square with 2*mu degrees of
freedom and noncentrality 2*mu*K*zeta*(1+Delta*cos(theta)).
Population normalization preserves sample independence.

Organization and outputs
------------------------
Simulator; PDF/CDF reconstruction; bounded optimization; figures and CLI.
Fit the trapezoidal CDF; retain the guide's cumsum-gradient CDF for comparison.
D_KS is the largest CDF gap on the supplied grid, not a calibrated KS test.
N_min=(1.36/D_KS)**2 is a hypothetical classical-KS size benchmark, NOT the
path count or evidence of significance. Peak counts are descriptive only.
A fixed simulation seed makes objective calls repeatable. The optimizer has a
separate seed that defaults to the simulation seed. Fresh-seed checks measure
Monte Carlo sensitivity, not measurement uncertainty or fit optimality.

Run from repository root:
  .venv/bin/python scripts/fit_curvas.py data/scenarios/scenario_*/pdf_data.csv
  .venv/bin/python scripts/fit_curvas.py data/scenarios/scenario_*/pdf_data.csv --diagnostics-only

Writes results/residuo_curvas/<scenario>/CDF comparison CSV/PNG and, when
fitting, fit_results.json and residuo.png; also run_config.json and summary.csv.
Diagnostics-only skips the expensive optimization. CLI exposes search bounds,
mu grid, simulation size, iteration/population settings, both seeds and output folder.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks

SEED, NSIM = 4242, 200_000
DEFAULT_BOUNDS = ((0.0, 50.0), (0.0, 1.0), (0.1, 60.0))
DEFAULT_MU_GRID = (1, 2, 3, 4, 5)


def sample_mftr(K, Delta, m, mu, n_samples=NSIM, seed=SEED):
    """Draw independent envelopes with population E[R**2]=1.

    K>=0 is total specular/diffuse mean power; Delta in [0,1] is
    2*V1*V2 / total specular power; real m>0 gives Var(zeta)=1/m;
    mu must be a positive integer, never rounded.
    With sigma**2=1: P=2*mu*K, V1**2+V2**2=P, 2*V1*V2=Delta*P.
    Additional diffuse clusters contribute Gamma(mu-1,scale=2) power.
    Circular Gaussian symmetry lets the first ray's phase be set to zero.
    Divide power by its theoretical mean 2*mu*(1+K), not the sample mean.
    For K=0, R**2 is Gamma(mu,scale=1/mu), independently of Delta and m.
    """
    if not np.all(np.isfinite([K, Delta, m, mu])):
        raise ValueError("MFTR parameters must be finite")
    if K < 0 or not 0 <= Delta <= 1 or m <= 0:
        raise ValueError("Require K>=0, 0<=Delta<=1 and m>0")
    if mu < 1 or int(mu) != mu:
        raise ValueError("This experiment requires integer mu>=1")
    if n_samples < 1 or int(n_samples) != n_samples:
        raise ValueError("n_samples must be a positive integer")
    
    mu, n_samples = int(mu), int(n_samples)
    rng = np.random.default_rng(seed)
    P = 2.0 * mu * K
    balance = np.sqrt(max(0.0, 1.0 - Delta**2))
    v1 = np.sqrt(P * (1.0 + balance) / 2.0)
    v2 = np.sqrt(P * (1.0 - balance) / 2.0)
    fluctuation = np.sqrt(rng.gamma(m, 1.0 / m, n_samples))
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    real = fluctuation * (v1 + v2 * np.cos(theta)) + rng.normal(size=n_samples)
    imag = fluctuation * v2 * np.sin(theta) + rng.normal(size=n_samples)
    power = real**2 + imag**2
    if mu > 1:
        power += rng.gamma(mu - 1, 2.0, n_samples)
    return np.sqrt(power / (2.0 * mu * (1.0 + K)))


def load_pdf(path, *, remove_exact_duplicates=False):
    """Validate/normalize an envelope PDF and return both reconstructed CDFs.

    Sort xi; require unique finite xi>=0 and finite nonnegative PDF values.
    Assume a piecewise linear PDF on the supplied interval, zero outside.
    Compute area by trapecios; integrate x**2*p(x) exactly under that same
    interpolation using two-point Gaussian quadrature on each cubic interval.
    For a=sqrt(E[x**2]), return normalized x/a and density a*p(x).

    The trapezoidal CDF starts at zero and integrates intervals. The guide's
    cumsum(p*gradient(x)) instead assigns weights to points; its first value
    can be positive and endpoint weights differ. Normalize each to end at one.
    Return arrays, original area and original second moment in a dictionary.
    Neither method recovers unobserved tails or measurement uncertainty.
    """
    data = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
    if not {"xi", "pdf_empirical"}.issubset(data.dtype.names or ()):
        raise ValueError(f"{path}: require columns xi,pdf_empirical")
    raw_x, raw_p = data["xi"], data["pdf_empirical"]
    if remove_exact_duplicates:
        points = np.c_[raw_x, raw_p]
        unique, first, counts = np.unique(
            points, axis=0, return_index=True, return_counts=True
        )
        removed = int(np.sum(counts - 1))
        removed_pairs = unique[counts > 1].tolist()
        raw_x, raw_p = raw_x[first], raw_p[first]
    else:
        removed = 0
        removed_pairs = []
    order = np.argsort(raw_x)
    x, p = raw_x[order], raw_p[order]
    if len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        raise ValueError(f"{path}: need at least two finite XY points")
    if np.any(x < 0) or np.any(p < 0) or np.any(np.diff(x) <= 0):
        raise ValueError(f"{path}: require unique xi>=0 and pdf>=0")
    area = np.trapezoid(p, x)
    if area <= 0:
        raise ValueError(f"{path}: PDF must have positive area")
    p = p / area
    width = np.diff(x)
    t = np.array([0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)])
    nodes = x[:-1, None] + width[:, None] * t
    values = p[:-1, None] + np.diff(p)[:, None] * t
    omega = float(np.sum(width * np.mean(nodes**2 * values, axis=1)))
    if omega <= 0:
        raise ValueError(f"{path}: second moment must be positive")
    x, p = x / np.sqrt(omega), p * np.sqrt(omega)
    trap = cumulative_trapezoid(p, x, initial=0)
    trap /= trap[-1]
    guide = np.cumsum(p * np.gradient(x))
    guide /= guide[-1]
    return dict(
        xi=x,
        pdf=p,
        cdf_trapezoid=trap,
        cdf_guide=guide,
        original_area=float(area),
        original_omega=omega,
        exact_duplicate_rows_removed=int(removed),
        exact_duplicate_xy_removed=removed_pairs,
        input_point_count=int(len(data)),
        fit_point_count=int(len(x)),
    )


def sample_cdf(samples, xi):
    """Compute P(samples<=xi) without sorting the entire simulated sample.

    Histogram all values, including those outside the empirical support.
    Advance bin edges by one floating-point step to include equality with xi.
    This avoids a repeated full sort inside differential evolution.
    """
    edges = np.r_[-np.inf, np.nextafter(xi, np.inf), np.inf]
    counts, _ = np.histogram(samples, bins=edges)
    return np.cumsum(counts[:-1]) / len(samples)


def fit_curve(
    xi,
    cdf_emp,
    mu_grid=DEFAULT_MU_GRID,
    *,
    bounds=DEFAULT_BOUNDS,
    n_sim=NSIM,
    seed=SEED,
    optimizer_seed=None,
    maxiter=40,
    popsize=15,
    polish=False,
):
    """Fit every integer mu and return the best candidate plus all candidates.

    Bounds follow K,Delta,m. Record optimizer success/message, evaluations and
    parameters within 1% of a boundary. A fixed simulation seed reduces random
    changes between calls, but the finite-sample CDF objective is stepped:
    gradient polishing is optional and off by default.
    A best candidate is not proof of a global optimum; exhausted iterations
    and proximity to bounds must remain visible in a scientific report.
    """
    candidates = []
    if optimizer_seed is None:
        optimizer_seed = seed
    for mu in mu_grid:

        def objective(params):
            samples = sample_mftr(*params, mu, n_samples=n_sim, seed=seed)
            return float(np.max(np.abs(cdf_emp - sample_cdf(samples, xi))))

        result = differential_evolution(
            objective,
            bounds,
            seed=optimizer_seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=polish,
            workers=1,
            updating="deferred",
        )
        boundary = [
            name
            for name, value, (lo, hi) in zip(("K", "Delta", "m"), result.x, bounds)
            if min(value - lo, hi - value) <= 0.01 * (hi - lo)
        ]
        candidate = dict(
            K=float(result.x[0]),
            Delta=float(result.x[1]),
            m=float(result.x[2]),
            mu=int(mu),
            D_KS=float(result.fun),
            success=bool(result.success),
            message=str(result.message),
            nfev=int(result.nfev),
            optimizer_seed=int(optimizer_seed),
            near_boundary=boundary,
        )
        candidates.append(candidate)
        print(
            f"  mu={mu} D_grid={result.fun:.5f} converged={result.success} boundary={boundary}",
            flush=True,
        )
    return min(candidates, key=lambda c: c["D_KS"]), candidates


def save_cdf_comparison(curve, destination):
    """Export both CDFs and their signed difference to CSV and a two-panel PNG.

    Their difference measures integration-method sensitivity, not noise.
    These outputs do not require a fit or any empirical PDF resampling.
    """
    x, trap, guide = curve["xi"], curve["cdf_trapezoid"], curve["cdf_guide"]
    np.savetxt(
        destination / "cdf_comparison.csv",
        np.c_[x, curve["pdf"], trap, guide, guide - trap],
        delimiter=",",
        header="xi_normalized,pdf_normalized,cdf_trapezoid,cdf_guide,guide_minus_trapezoid",
        comments="",
    )
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(x, trap, label="Trapecios", color="#0072B2")
    ax[0].plot(x, guide, "--", label="Guia: cumsum-gradient", color="#D55E00")
    ax[0].set_ylabel("CDF")
    ax[0].legend()
    ax[1].plot(x, guide - trap, color="#D55E00")
    ax[1].axhline(0, color="black", lw=0.7)
    ax[1].set_ylabel("CDF guia - CDF trapecios")
    for axis in ax:
        axis.set_xlabel("Envolvente normalizada")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "cdf_comparison.png", dpi=180)
    plt.close(fig)


def save_fit_plot(curve, samples, mu, destination):
    """Plot the PDF, trapezoidal CDF residual and lower tail descriptively.

    Histogram heights divide by full simulation size, preserving out-of-view
    mass. On log axes mask zero CDF values instead of inventing tail probability.
    """
    x, p = curve["xi"], curve["pdf"]
    model_cdf = sample_cdf(samples, x)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    edges = np.linspace(0, x[-1], 120)
    counts, _ = np.histogram(samples, bins=edges)
    ax[0].plot(x, p, label="PDF reconstruida", color="#0072B2")
    ax[0].plot(
        (edges[:-1] + edges[1:]) / 2,
        counts / (len(samples) * np.diff(edges)),
        "--",
        label=f"MFTR (mu={mu})",
        color="#D55E00",
    )
    ax[0].set_ylabel("PDF")
    ax[0].set_title("(a) Distribuciones")
    empirical_cdf = curve["cdf_trapezoid"]
    ax[1].plot(
        x,
        empirical_cdf - model_cdf,
        "-",
        label="CDF empírica − MFTR",
        color="#0072B2",
    )
    ax[1].axhline(0, color="black", lw=0.7)
    ax[1].set_ylabel("CDF empírica − CDF MFTR")
    ax[1].set_title("(b) Discrepancia descriptiva")
    ax[2].semilogy(
        x,
        np.ma.masked_less_equal(empirical_cdf, 0),
        label="CDF empírica",
        color="#0072B2",
    )
    ax[2].semilogy(
        x, np.ma.masked_less_equal(model_cdf, 0), label="MFTR", color="#D55E00"
    )
    ax[2].set_ylabel("CDF")
    ax[2].set_title("(c) Cola inferior")
    for axis in ax:
        axis.set_xlabel("Envolvente normalizada")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(destination / "residuo.png", dpi=180)
    plt.close(fig)


def main():
    """Run diagnostics or fits, saving assumptions, candidates and summary.

    Fits compare both integrations at the same fitted parameters, then repeat
    simulation with three fresh seeds. This is Monte Carlo sensitivity only.
    Repeating a fit with another optimizer seed is supported for stability
    checks. The script does not independently refit to the guide CDF.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/residuo_curvas")
    )
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--n-sim", type=int, default=NSIM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--optimizer-seed", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--popsize", type=int, default=15)
    parser.add_argument("--mu-grid", type=int, nargs="+", default=list(DEFAULT_MU_GRID))
    parser.add_argument(
        "--k-bounds", type=float, nargs=2, default=list(DEFAULT_BOUNDS[0])
    )
    parser.add_argument(
        "--m-bounds", type=float, nargs=2, default=list(DEFAULT_BOUNDS[2])
    )
    parser.add_argument("--polish", action="store_true")
    args = parser.parse_args()
    if args.optimizer_seed is None:
        args.optimizer_seed = args.seed
    bounds = (tuple(args.k_bounds), (0.0, 1.0), tuple(args.m_bounds))
    if (
        args.n_sim < 1
        or args.maxiter < 1
        or args.popsize < 1
        or any(mu not in DEFAULT_MU_GRID for mu in args.mu_grid)
        or not all(np.isfinite([lo, hi]).all() and lo < hi for lo, hi in bounds)
        or bounds[0][0] < 0
        or bounds[0][1] > 50
        or bounds[2][0] <= 0
        or bounds[2][0] < 0.1
        or bounds[2][1] > 60
    ):
        parser.error("Require positive counts, mu in 1..5, K in [0,50], m in [0.1,60]")
    labels = [p.parent.name if p.name == "pdf_data.csv" else p.stem for p in args.paths]
    if len(set(labels)) != len(labels):
        parser.error("Input labels must be unique to avoid overwriting outputs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k != "paths"
    }
    config.update(
        paths=[str(p) for p in args.paths],
        simulation_seed=args.seed,
        optimizer_seed=args.optimizer_seed,
        bounds_K_Delta_m=bounds,
        fitting_cdf="trapezoid",
        omega=1.0,
        missing_tail_mass="assumed zero",
        N_min_interpretation="hypothetical KS benchmark; not calibrated significance",
        empirical_sampling=False,
        sampler="sqrt-Gamma, U_i=0, theoretical normalization",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    rows = []
    for path, label in zip(args.paths, labels):
        print(f"\n=== {label} ===", flush=True)
        destination = args.output_dir / label
        destination.mkdir(parents=True, exist_ok=True)
        curve = load_pdf(path, remove_exact_duplicates=True)
        save_cdf_comparison(curve, destination)
        peaks, _ = find_peaks(curve["pdf"], height=0.15 * curve["pdf"].max())
        cdf_gap = float(np.max(np.abs(curve["cdf_guide"] - curve["cdf_trapezoid"])))
        row = dict(
            scenario=label,
            original_area=curve["original_area"],
            original_omega=curve["original_omega"],
            input_point_count=curve["input_point_count"],
            fit_point_count=curve["fit_point_count"],
            exact_duplicate_rows_removed=curve["exact_duplicate_rows_removed"],
            peaks_height_15pct=len(peaks),
            max_cdf_integration_gap=cdf_gap,
        )
        print(
            f"  max |CDF guia - trapecios|={cdf_gap:.6f}; peaks={len(peaks)}",
            flush=True,
        )
        if not args.diagnostics_only:
            best, candidates = fit_curve(
                curve["xi"],
                curve["cdf_trapezoid"],
                args.mu_grid,
                bounds=bounds,
                n_sim=args.n_sim,
                seed=args.seed,
                optimizer_seed=args.optimizer_seed,
                maxiter=args.maxiter,
                popsize=args.popsize,
                polish=args.polish,
            )
            params = [best[k] for k in ("K", "Delta", "m", "mu")]
            samples = sample_mftr(*params, n_samples=args.n_sim, seed=args.seed)
            model_cdf = sample_cdf(samples, curve["xi"])
            diff = curve["cdf_trapezoid"] - model_cdf
            i = int(np.argmax(np.abs(diff)))
            fresh_gaps = [
                float(
                    np.max(
                        np.abs(
                            curve["cdf_trapezoid"]
                            - sample_cdf(
                                sample_mftr(
                                    *params, n_samples=args.n_sim, seed=args.seed + j
                                ),
                                curve["xi"],
                            )
                        )
                    )
                )
                for j in (1, 2, 3)
            ]
            row.update(
                {k: best[k] for k in ("K", "Delta", "m", "mu", "D_KS", "success")}
            )
            row.update(
                D_KS_guide_same_fit=float(
                    np.max(np.abs(curve["cdf_guide"] - model_cdf))
                ),
                max_gap_percentile=100 * float(curve["cdf_trapezoid"][i]),
                max_gap_xi=float(curve["xi"][i]),
                signed_max_gap=float(diff[i]),
                N_min_hypothetical=(1.36 / best["D_KS"]) ** 2 if best["D_KS"] else None,
                fresh_seed_gap_min=min(fresh_gaps),
                fresh_seed_gap_max=max(fresh_gaps),
                near_boundary=",".join(best["near_boundary"]),
            )
            (destination / "fit_results.json").write_text(
                json.dumps(
                    dict(
                        summary=row,
                        simulation_seed=args.seed,
                        optimizer_seed=args.optimizer_seed,
                        input_path=str(path),
                        preparation={
                            "input_point_count": curve["input_point_count"],
                            "fit_point_count": curve["fit_point_count"],
                            "exact_duplicate_rows_removed": curve[
                                "exact_duplicate_rows_removed"
                            ],
                            "exact_duplicate_xy_removed": curve[
                                "exact_duplicate_xy_removed"
                            ],
                        },
                        candidates=candidates,
                        fresh_seed_gaps=fresh_gaps,
                    ),
                    indent=2,
                )
                + "\n"
            )
            save_fit_plot(curve, samples, best["mu"], destination)
        rows.append(row)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"\nResults: {args.output_dir}. Descriptive gaps; significance not established."
    )


if __name__ == "__main__":
    main()
