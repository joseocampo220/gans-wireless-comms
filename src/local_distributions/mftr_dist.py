import numpy as np
from scipy.special import gamma
from scipy.stats import rv_continuous
from typing import Optional
import math

"""
This code was adapted from the original MATLAB implementation of the MFTR distribution.
The original code can be found at: https://github.com/JoseDavidVega/MFTR-Fading-Channel-Model
"""

# -----------------------
# Utilities / settings
# -----------------------

DEFAULT_INV_TERMS = (
    4000  # trade-off between accuracy and speed (was 1e4 in original MATLAB)
)
A_INV = 15.0  # parameter used in inverse Laplace (kept from original)

# -----------------------
# Vectorized low-level functions
# -----------------------


def _laplace_phi2(
    s: np.ndarray, lam: float, delta: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """
    Vectorized LaplacePhi2: accepts s with arbitrary shape (complex dtype),
    delta and beta are 1-D arrays (length K).
    Returns same shape as s.
    """
    s = np.asarray(s, dtype=np.complex128)
    P = np.ones_like(s, dtype=np.complex128)
    # delta and beta are small (length 4 in model)
    for d, b in zip(delta, beta):
        P *= (1.0 - (d / s)) ** (-b)
    return gamma(lam) * (s ** (-lam)) * P


def _phi2(
    x: np.ndarray,
    lam: float,
    delta: np.ndarray,
    beta: np.ndarray,
    n_terms: int = DEFAULT_INV_TERMS,
) -> np.ndarray:
    """
    Vectorized Phi2: inverse Laplace sum evaluated for a vector x.
    x: 1-D array of strictly positive reals.
    Returns array of same shape as x.
    Note: memory cost ~ O(n_terms * len(x)); reduce n_terms if RAM is tight.
    """
    x = np.asarray(x, dtype=float).ravel()
    if np.any(x <= 0.0):
        raise ValueError("x must be strictly positive for _phi2")

    # build s matrix shape (n_terms, len(x))
    n = np.arange(1, n_terms + 1)
    # s_n = (A + 2j*pi*(n-1)) * (1 / (2 * x))
    # We'll compute outer product-like: factor_n[:,None] * (1/(2*x))[None,:]
    factor_n = A_INV + 2j * np.pi * (n - 1)  # shape (n_terms,)
    inv_2x = 1.0 / (2.0 * x)  # shape (len(x),)
    s = factor_n[:, None] * inv_2x[None, :]  # shape (n_terms, len(x)), complex

    # alphainv vector
    alphainv = np.ones(n_terms, dtype=float)
    alphainv[0] = 0.5

    # Evaluate LaplacePhi2 over s (vectorized)
    # returns shape (n_terms, len(x))
    L = _laplace_phi2(s, lam, delta, beta)  # complex
    # Only real part is used in sum
    terms = ((-1.0) ** (n - 1))[:, None] * alphainv[:, None] * np.real(L)
    y1 = np.sum(terms, axis=0)  # shape (len(x),)

    kfac = np.exp(A_INV / 2.0) * (x**-1.0)
    y1 = kfac * y1
    y = (x ** (-lam + 1.0)) * y1
    return y


def pdf_mftr(
    power_vals: np.ndarray,
    m: float,
    K: float,
    delta: float,
    mu: float,
    omega: float,
    n_inverse_terms: int = DEFAULT_INV_TERMS,
) -> np.ndarray:
    """
    Vectorized MFTR PDF for POWER (gamma). power_vals can be array-like.
    Returns array with same shape as power_vals.
    """
    G = np.asarray(power_vals, dtype=float).ravel()
    a1 = m * mu * (1.0 + K) / (m + mu * K) / omega
    a2 = m * mu * (1.0 + K) / (m + mu * K * (1.0 + delta)) / omega
    a3 = m * mu * (1.0 + K) / (m + mu * K * (1.0 - delta)) / omega
    a4 = mu * (1.0 + K) / omega

    pdf = np.zeros_like(G, dtype=float)

    # precompute constants in log-space to avoid overflow
    log_base_prefactor = (m / 2.0) * np.log(a2 * a3) - (m - 1) * np.log(2.0) - (m - mu) * np.log(a4)
    base_prefactor = np.exp(log_base_prefactor)
    # number of q terms
    q_max = int(np.floor((m - 1) / 2))
    # loop over q (small)
    for q in range(0, q_max + 1):
        delta_vec = -np.array([a1, a2, a3, a4], dtype=float)
        beta_vec = np.array(
            [1 + 2 * q - m, m - q - 0.5, m - q - 0.5, mu - m], dtype=float
        )

        C = gamma(1 + 2 * (m - 1) - 2 * q) / (
            gamma(1 + q) * gamma(1 + m - 1 - q) * gamma(1 + m - 1 - 2 * q)
        )

        coef = (
            ((-1.0) ** q)
            * ((math.sqrt(a2 * a3) / a1) ** (m - 1 - 2 * q))
            * C
            * (1.0 / gamma(mu))
        )
        # we need Phi2 evaluated for all x=G
        # note: Phi2 expects scalar x in original code — our _phi2 handles vector x
        phi_vals = _phi2(G, mu, delta_vec, beta_vec, n_terms=n_inverse_terms)
        term = coef * (G ** (mu - 1.0)) * phi_vals
        pdf += term

    pdf = base_prefactor * pdf
    # keep original shape
    return pdf.reshape(np.shape(power_vals))


def cdf_mftr(
    power_vals: np.ndarray,
    m: float,
    K: float,
    delta: float,
    mu: float,
    omega: float,
    n_inverse_terms: int = DEFAULT_INV_TERMS,
) -> np.ndarray:
    """
    Vectorized MFTR CDF for POWER. Returns CDF(G).
    """
    G = np.asarray(power_vals, dtype=float).ravel()
    a1 = m * mu * (1.0 + K) / (m + mu * K) / omega
    a2 = m * mu * (1.0 + K) / (m + mu * K * (1.0 + delta)) / omega
    a3 = m * mu * (1.0 + K) / (m + mu * K * (1.0 - delta)) / omega
    a4 = mu * (1.0 + K) / omega

    cdf = np.zeros_like(G, dtype=float)
    base_prefactor = ((a2 * a3) ** (m / 2.0)) / (2.0 ** (m - 1) * (a4 ** (m - mu)))
    q_max = int(np.floor((m - 1) / 2))
    for q in range(0, q_max + 1):
        delta_vec = -np.array([a1, a2, a3, a4], dtype=float)
        beta_vec = np.array(
            [1 + 2 * q - m, m - q - 0.5, m - q - 0.5, mu - m], dtype=float
        )

        C = gamma(1 + 2 * (m - 1) - 2 * q) / (
            gamma(1 + q) * gamma(1 + m - 1 - q) * gamma(1 + m - 1 - 2 * q)
        )

        coef = (
            ((-1.0) ** q)
            * ((math.sqrt(a2 * a3) / a1) ** (m - 1 - 2 * q))
            * C
            * (1.0 / gamma(mu + 1.0))
        )
        phi_vals = _phi2(G, mu + 1.0, delta_vec, beta_vec, n_terms=n_inverse_terms)
        term = coef * (G**mu) * phi_vals
        cdf += term

    cdf = base_prefactor * cdf
    return cdf.reshape(np.shape(power_vals))


# -----------------------
# Fast physical-model sampler (vectorized)
# -----------------------


def gen_mftr_sim(
    m: float,
    mu: float,
    delta: float,
    K: float,
    n_samples: int,
    dist_type: str = "amplitude",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Fast vectorized sampler of normalized MFTR variates using the physical model.
    dist_type: "amplitude" or "power".
    Returns array shape (n_samples,)
    """

    rng = np.random.default_rng(seed)

    gb = 1.0
    mu_eff = float(mu)
    mu_clusters = int(max(1, round(mu_eff)))

    sig = np.sqrt(gb / np.sqrt(2.0 * mu_eff * (1.0 + K)))
    b = (sig**2) * mu_eff * K
    sqrt_term = math.sqrt(max(0.0, 1.0 - delta**2))
    v1 = math.sqrt(b * (1.0 + sqrt_term))
    v2 = math.sqrt(b * (1.0 - sqrt_term))

    # z ~ Nakagami(m, scale=1). We can sample via gamma: z^2 ~ Gamma(m, scale=1/m)
    # so z = sqrt(Gamma(...))
    m_eff = float(m)
    gam = rng.gamma(shape=m_eff, scale=1.0 / m_eff, size=n_samples)
    z = np.sqrt(gam)

    phi1 = rng.random(n_samples) * 2.0 * np.pi
    phi2 = rng.random(n_samples) * 2.0 * np.pi

    # sum of clusters (mu-1), vectorized:
    if mu_clusters - 1 > 0:
        # gaussian components with variance sig^2
        x_clusters = rng.normal(0.0, scale=sig, size=(mu_clusters - 1, n_samples))
        y_clusters = rng.normal(0.0, scale=sig, size=(mu_clusters - 1, n_samples))
        clus = np.sum(x_clusters**2 + y_clusters**2, axis=0)
    else:
        clus = np.zeros(n_samples)

    # last cluster
    X = rng.normal(0.0, scale=sig, size=n_samples)
    Y = rng.normal(0.0, scale=sig, size=n_samples)

    # resulting channel power (vectorized)
    term1 = (v1 * z * np.cos(phi1) + v2 * z * np.cos(phi2) + X) ** 2
    term2 = (v1 * z * np.sin(phi1) + v2 * z * np.sin(phi2) + Y) ** 2
    h = term1 + term2 + clus

    # normalization to unit average power
    denom = v1**2 + v2**2 + 2.0 * mu_eff * (sig**2)
    h = h / denom

    if dist_type == "amplitude":
        return np.sqrt(h)
    elif dist_type == "power":
        return np.abs(h)
    else:
        raise ValueError("dist_type must be 'amplitude' or 'power'")


class mftr(rv_continuous):
    """
    MFTR distribution for POWER (gamma).
    Analytic _pdf and _cdf use self.n_inverse_terms for accuracy/speed control.
    """

    def __init__(
        self, *args, n_inverse_terms=DEFAULT_INV_TERMS, dist_type="power", **kwargs
    ):
        self.n_inverse_terms = int(n_inverse_terms)
        self.dist_type = dist_type  # ← store here
        super().__init__(*args, **kwargs)

    def _argcheck(self, m, K, delta, mu, omega):
        # domain checks for shape params
        return (
            (m >= 1)
            and (mu >= 1)
            and (omega > 0)
            and (0.0 <= delta < 1.0)
            and (K >= 0.0)
        )

    # NOTE: no defaults in the signature other than x (required by scipy)
    def _pdf(self, x, m, K, delta, mu, omega):
        x = np.asarray(x)
        x_pos = np.maximum(x, np.finfo(float).tiny)
        return pdf_mftr(
            x_pos,
            float(m),
            float(K),
            float(delta),
            float(mu),
            float(omega),
            n_inverse_terms=self.n_inverse_terms,
        )

    def _cdf(self, x, m, K, delta, mu, omega):
        x = np.asarray(x)
        x_pos = np.maximum(x, 0.0)
        return cdf_mftr(
            x_pos,
            float(m),
            float(K),
            float(delta),
            float(mu),
            float(omega),
            n_inverse_terms=self.n_inverse_terms,
        )

    def _rvs(self, m, K, delta, mu, omega, size=None, random_state=None):
        # total samples
        n = int(np.prod(size)) if size is not None else 1

        # call your physical-model sampler
        samples = gen_mftr_sim(
            float(m),
            float(mu),
            float(delta),
            float(K),
            n,
            dist_type=self.dist_type,  # ← use stored dist_type
            seed=random_state,
        )

        # convert to amplitude if needed
        if self.dist_type == "amplitude":
            samples *= np.sqrt(omega)

        if size is None:
            return samples[0]

        return np.reshape(samples, size)
