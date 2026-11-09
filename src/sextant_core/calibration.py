"""Calibration utilities, uncertainty-quantification metrics, and bootstrap inference.

Provides:
- Simulation-based calibration ruler metrics (:func:`empirical_coverage`,
  :func:`expected_calibration_error`, :func:`relative_sharpness`, and
  :func:`evaluate_calibration`), explicitly scoped to synthetic or digital-reference
  regimes where ground-truth :math:`(D, D^*, f)` parameters exist.
- Non-parametric voxel-level bootstrap confidence intervals for boundary-railing
  fractions (:func:`bootstrap_fraction`, :func:`bootstrap_railing_ci`).
- Physiological :math:`(D, f)` context stratification across railed and interior
  voxels (:func:`summarize_df_context`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    from scipy.stats import norm as _scipy_norm
except ImportError:  # pragma: no cover - fallback when scipy is absent
    _scipy_norm = None

#: Default deterministic random seed across all Sextant analyses and benchmarks.
GLOBAL_SEED: int = 20260613

#: Default number of bootstrap resamples for 95% confidence intervals.
DEFAULT_BOOTSTRAP_SAMPLES: int = 5000

#: Default nominal central confidence levels for calibration evaluation.
DEFAULT_CALIBRATION_LEVELS: np.ndarray = np.array(
    [0.50, 0.68, 0.80, 0.90, 0.95], dtype=np.float64
)

_Z_BY_LEVEL: dict[float, float] = {
    0.50: 0.6744897501960817,
    0.68: 0.9944578832097530,
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.9599639845400540,
    0.99: 2.5758293035489004,
}


def make_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """Return a deterministic NumPy :class:`~numpy.random.Generator`.

    Parameters
    ----------
    seed : int, np.random.Generator, or None, optional
        Seed integer or existing generator. Defaults to :data:`GLOBAL_SEED`
        (``20260613``) when ``None``.

    Returns
    -------
    np.random.Generator
        Initialized random number generator.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(GLOBAL_SEED if seed is None else int(seed))


def _z_for_level(level: float) -> float:
    """Return the two-sided standard normal quantile for central coverage ``level``."""
    lv = float(level)
    if not (0.0 < lv < 1.0):
        raise ValueError(f"Confidence level must lie in (0, 1); got {level!r}.")
    for k, z_val in _Z_BY_LEVEL.items():
        if abs(lv - k) < 1e-6:
            return z_val
    if _scipy_norm is not None:
        return float(_scipy_norm.ppf(0.5 + 0.5 * lv))
    raise ValueError(f"Unsupported confidence level {level!r} without scipy.")


@dataclass(frozen=True)
class BootCI:
    """Container for a non-parametric percentile bootstrap confidence interval.

    Parameters
    ----------
    point : float
        Observed sample proportion or point estimate.
    lo : float
        Lower confidence bound at ``level``.
    hi : float
        Upper confidence bound at ``level``.
    level : float
        Nominal confidence level in :math:`(0, 1)` (e.g., ``0.95``).
    n_boot : int
        Number of bootstrap resamples drawn.
    seed : int
        Random seed used for reproducibility.
    """

    point: float
    lo: float
    hi: float
    level: float = 0.95
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES
    seed: int = GLOBAL_SEED

    def to_dict(self) -> dict[str, float | int]:
        """Serialize the bootstrap confidence interval to a dictionary."""
        return {
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "level": self.level,
            "n_boot": self.n_boot,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class CalibrationSummary:
    """Summary of Gaussian uncertainty calibration against synthetic ground truth.

    Parameters
    ----------
    coverage_by_level : dict of float to float
        Mapping from nominal coverage level :math:`\\alpha \\in (0, 1)` to empirical
        coverage fraction.
    ece : float
        Expected calibration error (mean absolute deviation between empirical
        and nominal coverage across evaluated levels).
    sharpness_rel : float
        Mean relative interval half-width at the 95% confidence level.
    n_samples : int
        Number of finite voxel estimates evaluated.
    """

    coverage_by_level: dict[float, float]
    ece: float
    sharpness_rel: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of calibration metrics."""
        return {
            "coverage_by_level": dict(self.coverage_by_level),
            "ece": self.ece,
            "sharpness_rel": self.sharpness_rel,
            "n_samples": self.n_samples,
        }


def requires_ground_truth() -> bool:
    """Return ``True``: calibration coverage/ECE require reference ground truth."""
    return True


def empirical_coverage(
    estimates: np.ndarray | Sequence[float],
    truth: np.ndarray | Sequence[float] | float,
    sigma: np.ndarray | Sequence[float] | float,
    levels: np.ndarray | Sequence[float] | None = None,
) -> dict[float, float]:
    """Compute empirical central Gaussian interval coverage across nominal levels.

    Parameters
    ----------
    estimates : np.ndarray or Sequence[float]
        Point estimates :math:`\\hat{\\theta}` of shape ``(N,)``.
    truth : np.ndarray, Sequence[float], or float
        Ground-truth parameter values :math:`\\theta^*` (scalar or shape ``(N,)``).
    sigma : np.ndarray, Sequence[float], or float
        Reported per-estimate standard deviations :math:`\\hat{\\sigma} > 0`.
    levels : np.ndarray or Sequence[float], optional
        Nominal confidence levels in :math:`(0, 1)`. Defaults to
        ``[0.50, 0.68, 0.80, 0.90, 0.95]``.

    Returns
    -------
    dict of float to float
        Dictionary mapping each nominal confidence level to empirical coverage.
    """
    est = np.asarray(estimates, dtype=np.float64)
    tru = np.asarray(truth, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    if est.size == 0:
        raise ValueError("`estimates` must be non-empty.")
    if np.any(sig <= 0) or not np.all(np.isfinite(sig)):
        raise ValueError("`sigma` must contain strictly positive finite values.")

    lvls = DEFAULT_CALIBRATION_LEVELS if levels is None else np.asarray(levels, dtype=np.float64)
    z_err = np.abs(est - tru) / sig
    return {
        float(lv): float(np.mean(z_err <= _z_for_level(float(lv))))
        for lv in lvls
    }


def expected_calibration_error(cov: dict[float, float]) -> float:
    """Compute Expected Calibration Error (ECE) from a coverage dictionary.

    .. math::

        \\text{ECE} = \\frac{1}{|\\mathcal{A}|} \\sum_{\\alpha \\in \\mathcal{A}}
        \\left| \\hat{c}(\\alpha) - \\alpha \\right|

    Parameters
    ----------
    cov : dict of float to float
        Mapping from nominal coverage :math:`\\alpha` to empirical coverage
        :math:`\\hat{c}(\\alpha)`.

    Returns
    -------
    float
        Mean absolute calibration gap across all evaluated levels.
    """
    if not cov:
        raise ValueError("`cov` dictionary must not be empty.")
    return float(np.mean([abs(float(emp) - float(nom)) for nom, emp in cov.items()]))


def relative_sharpness(
    sigma: np.ndarray | Sequence[float] | float,
    truth: np.ndarray | Sequence[float] | float,
    level: float = 0.95,
) -> float:
    """Compute mean relative confidence-interval half-width at ``level``.

    .. math::

        \\text{Sharpness}_{\\text{rel}}(\\alpha) = \\frac{1}{N} \\sum_{i=1}^N
        \\frac{z_{(1+\\alpha)/2} \\, \\hat{\\sigma}_i}{\\max(|\\theta^*_i|, 10^{-12})}

    Parameters
    ----------
    sigma : np.ndarray, Sequence[float], or float
        Per-voxel standard deviations :math:`\\hat{\\sigma}`.
    truth : np.ndarray, Sequence[float], or float
        Ground-truth parameter values :math:`\\theta^*`.
    level : float, default=0.95
        Nominal central confidence level in :math:`(0, 1)`.

    Returns
    -------
    float
        Mean relative interval half-width.
    """
    z = _z_for_level(level)
    sig_arr = np.asarray(sigma, dtype=np.float64)
    tru_arr = np.asarray(truth, dtype=np.float64)
    denom = np.maximum(np.abs(tru_arr), 1e-12)
    return float(np.mean(z * sig_arr / denom))


def evaluate_calibration(
    estimates: np.ndarray | Sequence[float],
    truth: np.ndarray | Sequence[float] | float,
    sigma: np.ndarray | Sequence[float] | float,
    levels: np.ndarray | Sequence[float] | None = None,
    sharpness_level: float = 0.95,
) -> CalibrationSummary:
    """Evaluate empirical coverage, ECE, and relative sharpness in one call.

    Parameters
    ----------
    estimates : np.ndarray or Sequence[float]
        Point estimates :math:`\\hat{\\theta}`.
    truth : np.ndarray, Sequence[float], or float
        Ground-truth parameter values :math:`\\theta^*`.
    sigma : np.ndarray, Sequence[float], or float
        Standard deviation estimates :math:`\\hat{\\sigma} > 0`.
    levels : np.ndarray or Sequence[float], optional
        Nominal confidence levels to evaluate.
    sharpness_level : float, default=0.95
        Confidence level at which to report relative interval half-width.

    Returns
    -------
    CalibrationSummary
        Structured calibration summary.
    """
    est = np.asarray(estimates, dtype=np.float64)
    cov = empirical_coverage(est, truth, sigma, levels=levels)
    ece_val = expected_calibration_error(cov)
    sharp_val = relative_sharpness(sigma, truth, level=sharpness_level)
    return CalibrationSummary(
        coverage_by_level=cov,
        ece=ece_val,
        sharpness_rel=sharp_val,
        n_samples=int(est.size),
    )


def bootstrap_fraction(
    railed_bool: np.ndarray | Sequence[bool],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    level: float = 0.95,
    seed: int = GLOBAL_SEED,
    chunk: int = 200,
) -> BootCI:
    """Compute a chunked percentile bootstrap CI for a boolean indicator array.

    Parameters
    ----------
    railed_bool : np.ndarray or Sequence[bool]
        1D boolean or binary array of per-voxel boundary-railing indicators.
    n_boot : int, default=5000
        Number of bootstrap resamples.
    level : float, default=0.95
        Confidence level in :math:`(0, 1)`.
    seed : int, default=20260613
        Deterministic seed for the random number generator.
    chunk : int, default=200
        Chunk size for bounded-memory resampling.

    Returns
    -------
    BootCI
        Percentile bootstrap confidence interval in :math:`[0, 1]`.
    """
    x = np.asarray(railed_bool, dtype=bool).astype(np.float64).ravel()
    n = x.size
    if n == 0:
        raise ValueError("`railed_bool` must not be empty.")
    if n_boot < 1:
        raise ValueError(f"`n_boot` must be >= 1; got {n_boot}.")
    if not (0.0 < level < 1.0):
        raise ValueError(f"`level` must lie in (0, 1); got {level}.")

    rng = make_rng(seed)
    point = float(x.mean())
    means = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done : done + m] = x[idx].mean(axis=1)
        done += m
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(means, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return BootCI(
        point=point,
        lo=float(lo),
        hi=float(hi),
        level=float(level),
        n_boot=int(n_boot),
        seed=int(seed),
    )


def bootstrap_railing_ci(
    railed_mask: np.ndarray | Sequence[bool],
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = GLOBAL_SEED,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute a parametric binomial bootstrap CI for the railing percentage (0-100%).

    Parameters
    ----------
    railed_mask : np.ndarray or Sequence[bool]
        Boolean mask of railed voxels.
    n_boot : int, default=5000
        Number of bootstrap draws.
    seed : int, default=20260613
        Random seed for reproducibility.
    level : float, default=0.95
        Confidence level in :math:`(0, 1)`.

    Returns
    -------
    point_pct : float
        Observed railing percentage in :math:`[0, 100]`.
    ci_low : float
        Lower bootstrap confidence bound in :math:`[0, 100]`.
    ci_high : float
        Upper bootstrap confidence bound in :math:`[0, 100]`.
    """
    arr = np.asarray(railed_mask, dtype=bool).ravel()
    if arr.size == 0:
        raise ValueError("`railed_mask` must not be empty.")
    rng = make_rng(seed)
    n = int(arr.size)
    point_prob = float(np.mean(arr))
    point_pct = 100.0 * point_prob
    k_samples = rng.binomial(n=n, p=point_prob, size=int(n_boot))
    boot_pcts = 100.0 * (k_samples / float(n))
    alpha = (1.0 - float(level)) / 2.0
    ci_low, ci_high = np.percentile(boot_pcts, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return point_pct, float(ci_low), float(ci_high)


def summarize_df_context(
    D_arr: np.ndarray | Sequence[float],
    f_arr: np.ndarray | Sequence[float],
    railed_mask: np.ndarray | Sequence[bool],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute median and interquartile range for :math:`D` and :math:`f` by railing status.

    Parameters
    ----------
    D_arr : np.ndarray or Sequence[float]
        Fitted tissue diffusivity array in :math:`\\text{mm}^2/\\text{s}`.
    f_arr : np.ndarray or Sequence[float]
        Fitted perfusion fraction array in :math:`[0, 1]`.
    railed_mask : np.ndarray or Sequence[bool]
        Boolean boundary-railing mask matching ``D_arr`` and ``f_arr``.

    Returns
    -------
    dict
        Nested dictionary keyed by ``"all"``, ``"railed"``, and ``"interior"``,
        reporting ``{"median", "q25", "q75"}`` for ``"D_1e3"``
        (:math:`\\times 10^{-3}\\,\\text{mm}^2/\\text{s}`) and ``"f_pct"`` (:math:`\\%`).
    """
    d = np.asarray(D_arr, dtype=np.float64)
    f = np.asarray(f_arr, dtype=np.float64)
    r = np.asarray(railed_mask, dtype=bool)

    def _med_iqr(arr: np.ndarray, scale: float) -> dict[str, float]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {"median": float("nan"), "q25": float("nan"), "q75": float("nan")}
        scaled = finite * scale
        q25, med, q75 = np.percentile(scaled, [25.0, 50.0, 75.0])
        return {"median": float(med), "q25": float(q25), "q75": float(q75)}

    non_railed = ~r
    return {
        "all": {
            "D_1e3": _med_iqr(d, 1e3),
            "f_pct": _med_iqr(f, 100.0),
        },
        "railed": {
            "D_1e3": _med_iqr(d[r], 1e3),
            "f_pct": _med_iqr(f[r], 100.0),
        },
        "interior": {
            "D_1e3": _med_iqr(d[non_railed], 1e3),
            "f_pct": _med_iqr(f[non_railed], 100.0),
        },
    }


# Backward-compatible short aliases matching sextant.ruler
coverage = empirical_coverage
ece = expected_calibration_error
sharpness_rel = relative_sharpness
