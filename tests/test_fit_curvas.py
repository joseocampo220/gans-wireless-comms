"""Scientific checks for the standalone MFTR curve-fitting script.

Load the script without executing its CLI. Test independently known laws:
zero specular power gives Gamma fading power; constant single-ray specular
power gives noncentral chi-square power. Also test population normalization,
exact moments of a piecewise linear PDF, CDF integration and input validation.
These checks validate the reference distribution and reconstruction, not the
existence of a measured THz residual or the optimizer's global optimality.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import gamma, ncx2

SCRIPT = Path(__file__).resolve().parents[1] / "data/scenarios/fit_curvas.py"
spec = importlib.util.spec_from_file_location("fit_curvas", SCRIPT)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


@pytest.mark.parametrize("mu", [1, 4])
def test_zero_specular_power_has_gamma_power_law(mu):
    """K=0 gives the known Gamma(mu,1/mu) law regardless of Delta and m."""
    samples = fit.sample_mftr(0, 0.8, 0.4, mu, 100_000, seed=71)
    grid = np.linspace(0.01, 2.5, 80)
    expected = gamma.cdf(grid**2, a=mu, scale=1 / mu)
    assert np.max(np.abs(fit.sample_cdf(samples, grid) - expected)) < 0.006


def test_no_shadowing_single_ray_has_noncentral_chi_square_power_law():
    """As m grows, zeta tends to one; Delta=0 removes phase interference."""
    K, mu = 4.0, 3
    samples = fit.sample_mftr(K, 0, 1_000_000, mu, 100_000, seed=19)
    grid = np.linspace(0.01, 2.0, 80)
    expected = ncx2.cdf(grid**2 * 2 * mu * (1 + K), df=2 * mu, nc=2 * mu * K)
    assert np.max(np.abs(fit.sample_cdf(samples, grid) - expected)) < 0.006


def test_population_unit_power_and_integer_domain():
    """Nonzero interference/shadowing retains unit expected power for mu>1."""
    samples = fit.sample_mftr(10, 0.7, 2.5, 3, 150_000, seed=31)
    assert abs(np.mean(samples**2) - 1) < 0.02
    with pytest.raises(ValueError, match="integer mu"):
        fit.sample_mftr(1, 0.5, 2, 1.4, 10)


def test_cdf_methods_against_uniform_density_and_exact_second_moment(tmp_path):
    """A uniform PDF exposes the guide's endpoint-weight error analytically."""
    path = tmp_path / "uniform.csv"
    path.write_text("xi,pdf_empirical\n0,0.5\n1,0.5\n2,0.5\n")
    curve = fit.load_pdf(path)
    assert curve["original_omega"] == pytest.approx(4 / 3)
    np.testing.assert_allclose(curve["cdf_trapezoid"], [0, 0.5, 1])
    np.testing.assert_allclose(curve["cdf_guide"], [1 / 3, 2 / 3, 1])
    np.testing.assert_allclose(curve["xi"], np.array([0, 1, 2]) / np.sqrt(4 / 3))


def test_histogram_cdf_counts_equality_and_outside_support():
    """Count exact grid ties and values outside support like a direct ECDF."""
    samples = np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 4.0])
    grid = np.array([0.0, 1.0, 2.0])
    expected = [(samples <= x).mean() for x in grid]
    np.testing.assert_allclose(fit.sample_cdf(samples, grid), expected)


def test_reject_duplicate_pdf_coordinates(tmp_path):
    """Duplicate xi cannot define a unique piecewise linear density."""
    path = tmp_path / "duplicate.csv"
    path.write_text("xi,pdf_empirical\n0,1\n1,1\n1,2\n")
    with pytest.raises(ValueError, match="unique"):
        fit.load_pdf(path)


def test_exact_duplicate_xy_rows_removed_only_when_requested(tmp_path):
    """Preparation removes repeated pairs in memory and records the count."""
    path = tmp_path / "duplicate_xy.csv"
    original = "xi,pdf_empirical\n0,1\n1,1\n1,1\n2,0\n"
    path.write_text(original)
    with pytest.raises(ValueError, match="unique"):
        fit.load_pdf(path)
    curve = fit.load_pdf(path, remove_exact_duplicates=True)
    assert curve["input_point_count"] == 4
    assert curve["fit_point_count"] == 3
    assert curve["exact_duplicate_rows_removed"] == 1
    assert curve["exact_duplicate_xy_removed"] == [[1.0, 1.0]]
    assert path.read_text() == original


def test_contradictory_density_at_same_coordinate_is_rejected(tmp_path):
    """Do not average or select densities when a coordinate repeats."""
    path = tmp_path / "contradictory.csv"
    path.write_text("xi,pdf_empirical\n0,1\n1,1\n1,2\n2,0\n")
    with pytest.raises(ValueError, match="unique"):
        fit.load_pdf(path, remove_exact_duplicates=True)
