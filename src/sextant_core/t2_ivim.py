"""2D (b, TE) relaxation-compensated IVIM fitting, CRLB, and boundary-railing diagnostics.

Implements the two-compartment diffusion-relaxation forward model, separable +
multi-start bounded non-linear least-squares (NLLS) fitting, exact analytical
5x5 Fisher Information Matrix / Cramér–Rao Lower Bound (CRLB) identifiability
analysis, and deterministic Monte Carlo comparison of 1D single-TE versus 2D
multi-echo ``(b, TE)`` boundary-railing.

The forward signal model is:

.. math::

    S(b, \\text{TE}) = S_0 \\left[
        f \\exp(-b D^*) \\exp(-\\text{TE} / T_{2,p})
        + (1 - f) \\exp(-b D) \\exp(-\\text{TE} / T_{2,t})
    \\right]

where:
- :math:`b` is the diffusion sensitization factor (:math:`\\text{s}/\\text{mm}^2`),
- :math:`\\text{TE}` is the echo time (:math:`\\text{ms}`),
- :math:`D` is the slow parenchymal tissue diffusivity (:math:`\\text{mm}^2/\\text{s}`),
- :math:`D^*` is the fast microvascular pseudo-diffusion coefficient (:math:`\\text{mm}^2/\\text{s}`),
- :math:`f \\in [0, 1]` is the relaxation-compensated microvascular fraction,
- :math:`T_{2,p}` is the microvascular transverse relaxation time (:math:`\\text{ms}`),
- :math:`T_{2,t}` is the parenchymal tissue transverse relaxation time (:math:`\\text{ms}`),
- :math:`S_0` is the unweighted (:math:`b = 0, \\text{TE} = 0`) proton-density baseline amplitude.

License: AGPL-3.0-or-later
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np

from .calibration import GLOBAL_SEED
from .fitting import (
    DEFAULT_B_THRESH,
    DEFAULT_BOUNDS_D,
    DEFAULT_BOUNDS_D_STAR,
    DEFAULT_BOUNDS_F,
    DEFAULT_RAIL_TOL,
    classify_railing,
    compute_crlb,
    fit_ivim,
)

try:
    from scipy.optimize import least_squares as _scipy_least_squares
except ImportError:  # pragma: no cover - fallback when scipy is absent
    _scipy_least_squares = None


# Default physiological transverse relaxation bounds (ms)
DEFAULT_BOUNDS_T2_P: tuple[float, float] = (20.0, 400.0)
DEFAULT_BOUNDS_T2_T: tuple[float, float] = (20.0, 250.0)

DEFAULT_T2_IVIM_BVALS: np.ndarray = np.array(
    [0.0, 15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 800.0],
    dtype=np.float64,
)
DEFAULT_T2_IVIM_TES: np.ndarray = np.array(
    [22.0, 42.0, 70.0, 110.0],
    dtype=np.float64,
)

PARAM_NAMES_5: tuple[str, str, str, str, str] = ("D", "D_star", "f", "T2_p", "T2_t")
PARAM_NAMES_6: tuple[str, str, str, str, str, str] = ("D", "D_star", "f", "T2_p", "T2_t", "S0")

DEFAULT_T2_IVIM_BOUNDS: dict[str, tuple[float, float]] = {
    "D": DEFAULT_BOUNDS_D,
    "D_star": DEFAULT_BOUNDS_D_STAR,
    "f": DEFAULT_BOUNDS_F,
    "T2_p": DEFAULT_BOUNDS_T2_P,
    "T2_t": DEFAULT_BOUNDS_T2_T,
}


def _make_pcg64_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """Create a deterministic PCG64 NumPy Generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    seed_val = GLOBAL_SEED if seed is None else int(seed)
    return np.random.Generator(np.random.PCG64(seed_val))


def _resolve_schedule(
    bvalues: np.ndarray | Sequence[float],
    te_values: np.ndarray | Sequence[float],
    grid: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], bool]:
    """Normalize ``(bvalues, te_values)`` into flattened paired vectors and schedule shape.

    Returns
    -------
    b_flat : np.ndarray of shape ``(M,)``
    te_flat : np.ndarray of shape ``(M,)``
    acq_shape : tuple of int, either ``(M,)`` (paired) or ``(N_b, N_te)`` (cartesian grid)
    is_grid : bool
    """
    b = np.asarray(bvalues, dtype=np.float64)
    te = np.asarray(te_values, dtype=np.float64)

    if b.size == 0 or te.size == 0:
        raise ValueError("`bvalues` and `te_values` must be non-empty.")
    if np.any(~np.isfinite(b)) or np.any(~np.isfinite(te)):
        raise ValueError("`bvalues` and `te_values` must contain finite numbers.")
    if np.any(b < 0.0):
        raise ValueError("`bvalues` must be non-negative (s/mm^2).")
    if np.any(te < 0.0):
        raise ValueError("`te_values` must be non-negative (ms).")

    if b.ndim == 2 or te.ndim == 2:
        if b.ndim > 2 or te.ndim > 2:
            raise ValueError(
                f"`bvalues` and `te_values` must be 1D or 2D arrays; got {b.shape} and {te.shape}."
            )
        b_grid, te_grid = np.broadcast_arrays(b, te)
        return (
            np.ascontiguousarray(b_grid.ravel(), dtype=np.float64),
            np.ascontiguousarray(te_grid.ravel(), dtype=np.float64),
            (int(b_grid.shape[0]), int(b_grid.shape[1])),
            True,
        )

    if b.ndim != 1 or te.ndim != 1:
        raise ValueError(
            f"`bvalues` and `te_values` must be 1D or 2D arrays; got {b.shape} and {te.shape}."
        )

    if grid is True or (grid is None and b.size != te.size):
        b_grid, te_grid = np.meshgrid(b, te, indexing="ij")
        return (
            np.ascontiguousarray(b_grid.ravel(), dtype=np.float64),
            np.ascontiguousarray(te_grid.ravel(), dtype=np.float64),
            (int(b.size), int(te.size)),
            True,
        )

    if b.size != te.size:
        raise ValueError(
            f"Paired 1D schedule requires equal lengths of `bvalues` ({b.size}) "
            f"and `te_values` ({te.size}), or pass `grid=True`."
        )
    return (
        np.ascontiguousarray(b, dtype=np.float64),
        np.ascontiguousarray(te, dtype=np.float64),
        (int(b.size),),
        False,
    )


def _normalize_bounds(
    bounds: Mapping[str, tuple[float, float]] | tuple[Sequence[float], Sequence[float]] | None,
    bounds_D: tuple[float, float] | None = None,
    bounds_D_star: tuple[float, float] | None = None,
    bounds_f: tuple[float, float] | None = None,
    bounds_T2_p: tuple[float, float] | None = None,
    bounds_T2_t: tuple[float, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """Parse flexible ``bounds`` specifications into a validated 5-parameter bounds dict."""
    out: dict[str, tuple[float, float]] = dict(DEFAULT_T2_IVIM_BOUNDS)
    if bounds is not None:
        if isinstance(bounds, Mapping):
            for k, v in bounds.items():
                if k in out:
                    out[k] = (float(v[0]), float(v[1]))
        elif isinstance(bounds, tuple) and len(bounds) == 2:
            lo_seq, hi_seq = bounds
            if len(lo_seq) not in (5, 6) or len(hi_seq) != len(lo_seq):
                raise ValueError(
                    "`bounds` tuple must contain lower and upper sequences of length 5 (or 6)."
                )
            for idx, name in enumerate(PARAM_NAMES_5):
                out[name] = (float(lo_seq[idx]), float(hi_seq[idx]))
        else:
            raise TypeError("`bounds` must be a dict or a 2-tuple `(lower, upper)`.")

    overrides = {
        "D": bounds_D,
        "D_star": bounds_D_star,
        "f": bounds_f,
        "T2_p": bounds_T2_p,
        "T2_t": bounds_T2_t,
    }
    for name, val in overrides.items():
        if val is not None:
            out[name] = (float(val[0]), float(val[1]))

    for name, (lo, hi) in out.items():
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"Invalid bounds for {name!r}: ({lo}, {hi}).")
    if out["T2_p"][0] <= 0.0 or out["T2_t"][0] <= 0.0:
        raise ValueError("Transverse relaxation bounds for `T2_p` and `T2_t` must be > 0 ms.")
    return out


def t2_ivim_forward_model(
    bvalues: np.ndarray | Sequence[float],
    te_values: np.ndarray | Sequence[float],
    D: np.ndarray | float,
    D_star: np.ndarray | float,
    f: np.ndarray | float,
    T2_p: np.ndarray | float,
    T2_t: np.ndarray | float,
    S0: np.ndarray | float = 1.0,
    *,
    grid: bool | None = None,
) -> np.ndarray:
    """Evaluate the 2D ``(b, TE)`` relaxation-compensated bi-exponential IVIM model.

    Computes:

    .. math::

        S(b, \\text{TE}) = S_0 \\left[
            f \\exp(-b D^*) \\exp(-\\text{TE} / T_{2,p})
            + (1 - f) \\exp(-b D) \\exp(-\\text{TE} / T_{2,t})
        \\right]

    Supports both 1D paired ``(b_i, TE_i)`` acquisition schedules of length ``M``
    and 2D cartesian ``(N_b, N_te)`` grids, as well as vectorized parameter arrays.

    Parameters
    ----------
    bvalues : np.ndarray or Sequence[float]
        Diffusion :math:`b`-values in :math:`\\text{s}/\\text{mm}^2` (1D paired of
        length ``M``, 1D axis of length ``N_b``, or 2D grid of shape ``(N_b, N_te)``).
    te_values : np.ndarray or Sequence[float]
        Echo times :math:`\\text{TE}` in :math:`\\text{ms}` (1D paired of length ``M``,
        1D axis of length ``N_te``, or 2D grid of shape ``(N_b, N_te)``).
    D : np.ndarray or float
        Slow tissue diffusivity :math:`D` (:math:`\\text{mm}^2/\\text{s}`).
    D_star : np.ndarray or float
        Fast pseudo-diffusion coefficient :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
    f : np.ndarray or float
        Relaxation-compensated microvascular fraction :math:`f \\in [0, 1]`.
    T2_p : np.ndarray or float
        Microvascular transverse relaxation time :math:`T_{2,p}` (:math:`\\text{ms}`).
    T2_t : np.ndarray or float
        Parenchymal tissue transverse relaxation time :math:`T_{2,t}` (:math:`\\text{ms}`).
    S0 : np.ndarray or float, default=1.0
        Unweighted (:math:`b = 0, \\text{TE} = 0`) baseline signal intensity :math:`S_0`.
    grid : bool or None, optional
        If ``True``, evaluate on the 2D cartesian product ``(len(bvalues), len(te_values))``
        even when ``len(bvalues) == len(te_values)``. If ``False``, require a 1D paired
        schedule. When ``None`` (default), 1D inputs of unequal length or 2D inputs
        produce a ``(N_b, N_te)`` grid, while equal-length 1D inputs are paired ``(M,)``.

    Returns
    -------
    np.ndarray
        Simulated signal array of shape ``(*param_shape, M)`` for paired schedules
        or ``(*param_shape, N_b, N_te)`` for cartesian grids.
    """
    b_flat, te_flat, acq_shape, _ = _resolve_schedule(bvalues, te_values, grid=grid)

    d_in = np.asarray(D, dtype=np.float64)
    ds_in = np.asarray(D_star, dtype=np.float64)
    f_in = np.asarray(f, dtype=np.float64)
    t2p_in = np.asarray(T2_p, dtype=np.float64)
    t2t_in = np.asarray(T2_t, dtype=np.float64)
    s0_in = np.asarray(S0, dtype=np.float64)

    param_shape = np.broadcast_shapes(
        d_in.shape, ds_in.shape, f_in.shape, t2p_in.shape, t2t_in.shape, s0_in.shape
    )

    d_arr = d_in[..., None]
    ds_arr = ds_in[..., None]
    f_arr = f_in[..., None]
    t2p_arr = t2p_in[..., None]
    t2t_arr = t2t_in[..., None]
    s0_arr = s0_in[..., None]

    ep = np.exp(-b_flat * ds_arr - te_flat / t2p_arr)
    et = np.exp(-b_flat * d_arr - te_flat / t2t_arr)
    sig_flat = s0_arr * (f_arr * ep + (1.0 - f_arr) * et)
    return sig_flat.reshape(param_shape + acq_shape)


@dataclass
class T2IVIMFitResult(Mapping[str, Any]):
    """Structured result container for 2D ``(b, TE)`` relaxation-compensated IVIM fits.

    Parameters
    ----------
    D : np.ndarray or float
        Fitted slow parenchymal diffusivity :math:`D` (:math:`\\text{mm}^2/\\text{s}`).
    D_star : np.ndarray or float
        Fitted microvascular pseudo-diffusivity :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
    f : np.ndarray or float
        Fitted relaxation-compensated perfusion fraction :math:`f \\in [0, 1]`.
    T2_p : np.ndarray or float
        Fitted microvascular relaxation time :math:`T_{2,p}` (:math:`\\text{ms}`).
    T2_t : np.ndarray or float
        Fitted parenchymal relaxation time :math:`T_{2,t}` (:math:`\\text{ms}`).
    S0 : np.ndarray or float
        Fitted baseline (:math:`b = 0, \\text{TE} = 0`) amplitude :math:`S_0`.
    railed_mask : np.ndarray or bool
        Boolean indicator(s) flagging :math:`D^*` boundary-railing within ``rail_tol``.
    railed_params : dict of str to (np.ndarray or bool)
        Per-parameter boundary-railing indicators across ``("D", "D_star", "f", "T2_p", "T2_t")``.
    rss : np.ndarray or float
        Residual sum of squares :math:`\\sum_i (\\hat{S}_i - y_i)^2`.
    success : np.ndarray or bool
        Convergence flag(s) for each fitted voxel.
    bounds : dict of str to tuple of (float, float)
        Box bounds applied to ``(D, D_star, f, T2_p, T2_t)``.
    lower_mask : np.ndarray or bool, default=False
        Lower-bound :math:`D^*` railing indicator(s).
    upper_mask : np.ndarray or bool, default=False
        Upper-bound :math:`D^*` railing indicator(s).
    affine : np.ndarray or None, optional
        Optional 4x4 spatial affine matrix for volumetric inputs.
    """

    D: np.ndarray | float
    D_star: np.ndarray | float
    f: np.ndarray | float
    T2_p: np.ndarray | float
    T2_t: np.ndarray | float
    S0: np.ndarray | float
    railed_mask: np.ndarray | bool
    railed_params: dict[str, np.ndarray | bool]
    rss: np.ndarray | float
    success: np.ndarray | bool
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    lower_mask: np.ndarray | bool = False
    upper_mask: np.ndarray | bool = False
    affine: np.ndarray | None = None

    @property
    def Dstar(self) -> np.ndarray | float:
        """Alias for :attr:`D_star`."""
        return self.D_star

    @property
    def dstar(self) -> np.ndarray | float:
        """Lowercase alias for :attr:`D_star`."""
        return self.D_star

    @property
    def railed(self) -> np.ndarray | bool:
        """Alias for :attr:`railed_mask`."""
        return self.railed_mask

    @property
    def frac_railed(self) -> float:
        """Fraction of finite fitted voxels whose :math:`D^*` estimate railed."""
        arr = np.asarray(self.D_star, dtype=np.float64)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return float("nan")
        r = np.asarray(self.railed_mask, dtype=bool)
        return float(np.mean(r[finite]))

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> list[str]:  # type: ignore[override]
        return [
            "D",
            "D_star",
            "f",
            "T2_p",
            "T2_t",
            "S0",
            "railed_mask",
            "railed_params",
            "rss",
            "success",
            "bounds",
        ]

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "D": self.D,
            "D_star": self.D_star,
            "Dstar": self.D_star,
            "dstar": self.D_star,
            "f": self.f,
            "T2_p": self.T2_p,
            "T2_t": self.T2_t,
            "S0": self.S0,
            "railed": self.railed_mask,
            "railed_mask": self.railed_mask,
            "railed_params": self.railed_params,
            "lower_mask": self.lower_mask,
            "upper_mask": self.upper_mask,
            "rss": self.rss,
            "success": self.success,
            "bounds": self.bounds,
        }
        if key in aliases:
            return aliases[key]
        raise KeyError(f"Unknown key {key!r} in T2IVIMFitResult")

    def to_dict(self) -> dict[str, Any]:
        """Convert fitted maps and diagnostics to a standard Python dictionary."""
        return {
            "D": self.D,
            "D_star": self.D_star,
            "f": self.f,
            "T2_p": self.T2_p,
            "T2_t": self.T2_t,
            "S0": self.S0,
            "railed_mask": self.railed_mask,
            "railed_params": self.railed_params,
            "lower_mask": self.lower_mask,
            "upper_mask": self.upper_mask,
            "frac_railed": self.frac_railed,
            "rss": self.rss,
            "success": self.success,
            "bounds": self.bounds,
        }


def _separable_warm_candidates(
    b_flat: np.ndarray,
    te_flat: np.ndarray,
    y_norm: np.ndarray,
    lo5: np.ndarray,
    hi5: np.ndarray,
    b_thresh: float,
) -> np.ndarray:
    """Build deterministic warm-start candidates ``(K, 5)`` ranked by variable-projected RSS."""
    # Step 1: High-b log-linear plane ln(y) ~ c0 - D*b - R2_t*TE for tissue (D, T2_t)
    pos_mask = y_norm > 1e-6
    high_mask = (b_flat >= b_thresh) & pos_mask
    if int(np.sum(high_mask)) < 3:
        order = np.argsort(b_flat)
        high_mask = np.zeros_like(b_flat, dtype=bool)
        high_mask[order[-min(max(3, b_flat.size // 2), b_flat.size) :]] = True
        high_mask &= pos_mask

    d_init = float(np.clip(1.1e-3, lo5[0], hi5[0]))
    t2t_init = float(np.clip(80.0, lo5[4], hi5[4]))
    a_t_init = 0.85

    if int(np.sum(high_mask)) >= 2:
        bh = b_flat[high_mask]
        teh = te_flat[high_mask]
        lyh = np.log(y_norm[high_mask])
        b_span = float(np.ptp(bh))
        te_span = float(np.ptp(teh))
        if b_span > 1e-6 and te_span > 1e-6 and bh.size >= 3:
            X_h = np.column_stack([np.ones_like(bh), -bh, -teh])
            beta, _, _, _ = np.linalg.lstsq(X_h, lyh, rcond=None)
            c0, d_raw, r2t_raw = float(beta[0]), float(beta[1]), float(beta[2])
            d_init = float(np.clip(d_raw, lo5[0], hi5[0]))
            if r2t_raw > 1e-5:
                t2t_init = float(np.clip(1.0 / r2t_raw, lo5[4], hi5[4]))
            a_t_init = float(np.clip(np.exp(c0), 1e-4, 20.0))
        elif b_span > 1e-6:
            X_h = np.column_stack([np.ones_like(bh), -bh])
            beta, _, _, _ = np.linalg.lstsq(X_h, lyh, rcond=None)
            d_init = float(np.clip(float(beta[1]), lo5[0], hi5[0]))
            # Estimate T2_t from lowest-b multi-TE slice if available
            b_min = float(np.min(b_flat))
            low_b = np.isclose(b_flat, b_min, atol=1.0) & pos_mask
            if int(np.sum(low_b)) >= 2 and float(np.ptp(te_flat[low_b])) > 1e-6:
                X_te = np.column_stack([np.ones(int(np.sum(low_b))), -te_flat[low_b]])
                beta_te, _, _, _ = np.linalg.lstsq(X_te, np.log(y_norm[low_b]), rcond=None)
                if beta_te[1] > 1e-5:
                    t2t_init = float(np.clip(1.0 / float(beta_te[1]), lo5[4], hi5[4]))
            a_t_init = float(
                np.clip(np.exp(float(beta[0]) + float(np.mean(teh)) / t2t_init), 1e-4, 20.0)
            )

    # Step 2: Residual microvascular log-linear plane on low-b points
    tissue_pred = a_t_init * np.exp(-b_flat * d_init - te_flat / t2t_init)
    res_perf = y_norm - tissue_pred
    low_perf_mask = (b_flat < max(b_thresh, 150.0)) & (res_perf > 0.015 * float(np.max(y_norm)))

    ds_init = float(np.clip(max(0.022, d_init * 3.0), lo5[1], hi5[1]))
    t2p_init = float(np.clip(t2t_init, lo5[3], hi5[3]))
    f_init = float(np.clip(0.15, lo5[2], hi5[2]))

    if int(np.sum(low_perf_mask)) >= 3:
        bl = b_flat[low_perf_mask]
        tel = te_flat[low_perf_mask]
        lyl = np.log(res_perf[low_perf_mask])
        if float(np.ptp(bl)) > 1e-6 and float(np.ptp(tel)) > 1e-6:
            X_l = np.column_stack([np.ones_like(bl), -bl, -tel])
            beta_l, _, _, _ = np.linalg.lstsq(X_l, lyl, rcond=None)
            a_p = float(np.clip(np.exp(float(beta_l[0])), 1e-5, 20.0))
            ds_raw = float(beta_l[1])
            r2p_raw = float(beta_l[2])
            ds_init = float(np.clip(max(ds_raw, d_init + 1e-4), lo5[1], hi5[1]))
            if r2p_raw > 1e-5:
                t2p_init = float(np.clip(1.0 / r2p_raw, lo5[3], hi5[3]))
            f_init = float(np.clip(a_p / max(a_p + a_t_init, 1e-9), lo5[2], hi5[2]))

    # Step 3: Deterministic multi-regime candidate set spanning T2_p < T2_t, T2_p ~ T2_t, T2_p > T2_t
    t2p_short = float(np.clip(0.45 * t2t_init, lo5[3], hi5[3]))
    t2p_mid = float(np.clip(0.95 * t2t_init, lo5[3], hi5[3]))
    t2p_long = float(np.clip(2.20 * t2t_init, lo5[3], hi5[3]))

    raw_candidates = [
        [d_init, ds_init, f_init, t2p_init, t2t_init],
        [d_init, 0.012, 0.12, t2p_short, t2t_init],
        [d_init, 0.025, 0.16, t2p_short, t2t_init],
        [d_init, 0.055, 0.20, t2p_short, t2t_init],
        [d_init, 0.015, 0.14, t2p_mid, t2t_init],
        [d_init, 0.035, 0.18, t2p_mid, t2t_init],
        [d_init, 0.012, 0.12, t2p_long, t2t_init],
        [d_init, 0.025, 0.16, t2p_long, t2t_init],
        [d_init, 0.055, 0.22, t2p_long, t2t_init],
        [d_init, 0.020, 0.05, t2p_short, t2t_init],
        [d_init, 0.020, 0.35, t2p_short, t2t_init],
        [d_init, 0.020, 0.35, t2p_long, t2t_init],
    ]
    cands = np.asarray(raw_candidates, dtype=np.float64)
    eps = 1e-7 * (hi5 - lo5)
    cands = np.clip(cands, lo5[None, :] + eps[None, :], hi5[None, :] - eps[None, :])
    cands[:, 1] = np.clip(
        np.maximum(cands[:, 1], cands[:, 0] + 1e-4), lo5[1] + eps[1], hi5[1] - eps[1]
    )

    # Vectorized Golub-Pereyra variable-projection scoring over all candidates
    ep = np.exp(-cands[:, 1:2] * b_flat[None, :] - te_flat[None, :] / cands[:, 3:4])
    et = np.exp(-cands[:, 0:1] * b_flat[None, :] - te_flat[None, :] / cands[:, 4:5])
    g = cands[:, 2:3] * ep + (1.0 - cands[:, 2:3]) * et  # (K, M)
    denom = np.sum(g * g, axis=1) + 1e-30
    s0_opt = np.maximum(np.sum(g * y_norm[None, :], axis=1) / denom, 1e-12)
    res = s0_opt[:, None] * g - y_norm[None, :]
    rss = np.sum(res * res, axis=1)
    order_idx = np.argsort(rss, kind="mergesort")
    return cands[order_idx]


def _solve_t2_ivim_6d(
    b_flat: np.ndarray,
    te_flat: np.ndarray,
    y_norm: np.ndarray,
    x0_6: np.ndarray,
    lo6: np.ndarray,
    hi6: np.ndarray,
    max_nfev: int = 350,
) -> tuple[np.ndarray, float, bool]:
    """Bounded 6-parameter NLLS solver with exact analytical Jacobian (TRF + pure-NumPy LM fallback)."""
    if _scipy_least_squares is not None:

        def residuals(p: np.ndarray) -> np.ndarray:
            d, ds, f_v, t2p, t2t, s0 = p[0], p[1], p[2], p[3], p[4], p[5]
            ep = np.exp(-b_flat * ds - te_flat / t2p)
            et = np.exp(-b_flat * d - te_flat / t2t)
            return s0 * (f_v * ep + (1.0 - f_v) * et) - y_norm

        def jacobian(p: np.ndarray) -> np.ndarray:
            d, ds, f_v, t2p, t2t, s0 = p[0], p[1], p[2], p[3], p[4], p[5]
            ep = np.exp(-b_flat * ds - te_flat / t2p)
            et = np.exp(-b_flat * d - te_flat / t2t)
            J = np.empty((b_flat.size, 6), dtype=np.float64)
            J[:, 0] = -b_flat * s0 * (1.0 - f_v) * et
            J[:, 1] = -b_flat * s0 * f_v * ep
            J[:, 2] = s0 * (ep - et)
            J[:, 3] = s0 * f_v * ep * (te_flat / (t2p * t2p))
            J[:, 4] = s0 * (1.0 - f_v) * et * (te_flat / (t2t * t2t))
            J[:, 5] = f_v * ep + (1.0 - f_v) * et
            return J

        x_start = np.clip(x0_6, lo6 + 1e-12, hi6 - 1e-12)
        try:
            res = _scipy_least_squares(
                residuals,
                x0=x_start,
                jac=jacobian,
                bounds=(lo6, hi6),
                method="trf",
                x_scale=[1e-3, 2e-2, 0.2, 60.0, 60.0, 1.0],
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=max_nfev,
            )
            r_final = res.fun
            return res.x.astype(np.float64), float(np.dot(r_final, r_final)), bool(res.status > 0)
        except Exception:
            pass

    # Pure-NumPy scaled projected Levenberg-Marquardt fallback
    scales = np.array([1e-3, 2e-2, 0.2, 60.0, 60.0, 1.0], dtype=np.float64)
    x = np.clip(np.asarray(x0_6, dtype=np.float64), lo6, hi6)
    lam = 1e-3
    eye6 = np.eye(6, dtype=np.float64)
    converged = False
    for _ in range(max_nfev):
        d, ds, f_v, t2p, t2t, s0 = x[0], x[1], x[2], x[3], x[4], x[5]
        ep = np.exp(-b_flat * ds - te_flat / t2p)
        et = np.exp(-b_flat * d - te_flat / t2t)
        r = s0 * (f_v * ep + (1.0 - f_v) * et) - y_norm
        cost = 0.5 * float(np.dot(r, r))

        J = np.empty((b_flat.size, 6), dtype=np.float64)
        J[:, 0] = -b_flat * s0 * (1.0 - f_v) * et
        J[:, 1] = -b_flat * s0 * f_v * ep
        J[:, 2] = s0 * (ep - et)
        J[:, 3] = s0 * f_v * ep * (te_flat / (t2p * t2p))
        J[:, 4] = s0 * (1.0 - f_v) * et * (te_flat / (t2t * t2t))
        J[:, 5] = f_v * ep + (1.0 - f_v) * et

        Js = J * scales[None, :]
        grad_s = Js.T @ r
        if float(np.linalg.norm(grad_s, ord=np.inf)) < 1e-9:
            converged = True
            break
        H_s = Js.T @ Js + lam * eye6
        try:
            step_s = np.linalg.solve(H_s, -grad_s)
        except np.linalg.LinAlgError:
            break
        x_new = np.clip(x + step_s * scales, lo6, hi6)
        if x_new[1] < x_new[0]:
            x_new[1] = min(max(x_new[0], lo6[1]), hi6[1])
        ep_n = np.exp(-b_flat * x_new[1] - te_flat / x_new[3])
        et_n = np.exp(-b_flat * x_new[0] - te_flat / x_new[4])
        r_new = x_new[5] * (x_new[2] * ep_n + (1.0 - x_new[2]) * et_n) - y_norm
        cost_new = 0.5 * float(np.dot(r_new, r_new))
        if cost_new < cost:
            if float(np.linalg.norm((x_new - x) / scales)) < 1e-9:
                x = x_new
                converged = True
                break
            x = x_new
            lam = max(lam * 0.3, 1e-12)
        else:
            lam *= 5.0
            if lam > 1e10:
                break

    ep_f = np.exp(-b_flat * x[1] - te_flat / x[3])
    et_f = np.exp(-b_flat * x[0] - te_flat / x[4])
    r_f = x[5] * (x[2] * ep_f + (1.0 - x[2]) * et_f) - y_norm
    return x, float(np.dot(r_f, r_f)), converged


def fit_t2_ivim_voxel(
    b_flat: np.ndarray,
    te_flat: np.ndarray,
    signal_flat: np.ndarray,
    lo5: np.ndarray,
    hi5: np.ndarray,
    b_thresh: float = DEFAULT_B_THRESH,
    max_nfev: int = 350,
    n_starts: int = 3,
) -> tuple[np.ndarray, float, bool]:
    """Fit a single voxel's 2D ``(b, TE)`` signal, returning ``([D, D_star, f, T2_p, T2_t, S0], rss, success)``."""
    if not np.all(np.isfinite(signal_flat)) or np.all(signal_flat <= 0.0):
        return np.full(6, np.nan, dtype=np.float64), float("nan"), False

    y_scale = float(np.max(signal_flat))
    if y_scale <= 0.0 or not np.isfinite(y_scale):
        return np.full(6, np.nan, dtype=np.float64), float("nan"), False

    y_norm = np.clip(signal_flat / y_scale, 0.0, None)

    # Upper bound on normalized S0: account for shortest sampled TE and minimum allowed T2
    te_min = float(np.min(te_flat))
    t2_min = min(float(lo5[3]), float(lo5[4]))
    s0_hi = max(50.0, 4.0 * float(np.exp(min(te_min / max(t2_min, 1.0), 6.0))))
    lo6 = np.concatenate([lo5, [1e-9]])
    hi6 = np.concatenate([hi5, [s0_hi]])

    ranked_cands = _separable_warm_candidates(b_flat, te_flat, y_norm, lo5, hi5, b_thresh=b_thresh)

    # Pick diverse top starts: best overall + best opposite T2_p/T2_t ordering
    chosen: list[np.ndarray] = [ranked_cands[0]]
    first_short = ranked_cands[0, 3] <= ranked_cands[0, 4]
    for cand in ranked_cands[1:]:
        is_short = cand[3] <= cand[4]
        if is_short != first_short:
            chosen.append(cand)
            break
    for cand in ranked_cands[1:]:
        if len(chosen) >= n_starts:
            break
        if not any(np.allclose(cand, prev, rtol=1e-3, atol=1e-5) for prev in chosen):
            chosen.append(cand)

    best_x6: np.ndarray | None = None
    best_rss_norm = float("inf")
    best_ok = False

    y_norm_sq = float(np.dot(y_norm, y_norm)) + 1e-30
    for idx, c5 in enumerate(chosen):
        ep0 = np.exp(-b_flat * c5[1] - te_flat / c5[3])
        et0 = np.exp(-b_flat * c5[0] - te_flat / c5[4])
        g0 = c5[2] * ep0 + (1.0 - c5[2]) * et0
        s0_0 = float(np.clip(np.dot(g0, y_norm) / (np.dot(g0, g0) + 1e-30), 1e-6, s0_hi * 0.9))
        x0_6 = np.concatenate([c5, [s0_0]])

        x_est, rss_n, ok = _solve_t2_ivim_6d(
            b_flat=b_flat,
            te_flat=te_flat,
            y_norm=y_norm,
            x0_6=x0_6,
            lo6=lo6,
            hi6=hi6,
            max_nfev=max_nfev,
        )

        # Enforce strict compartment ordering D <= D_star
        if x_est[0] > x_est[1]:
            lo6_ord = lo6.copy()
            lo6_ord[1] = min(max(lo6[1], x_est[0] + 1e-6), hi6[1] - 1e-7)
            x_re = x_est.copy()
            x_re[1] = max(x_re[1], lo6_ord[1] + 1e-7)
            x_est, rss_n, ok = _solve_t2_ivim_6d(
                b_flat=b_flat,
                te_flat=te_flat,
                y_norm=y_norm,
                x0_6=x_re,
                lo6=lo6_ord,
                hi6=hi6,
                max_nfev=max_nfev,
            )

        if rss_n < best_rss_norm:
            best_rss_norm = rss_n
            best_x6 = x_est
            best_ok = ok

        # Early exit on near-noiseless exact recovery after checking opposite T2 ordering
        if idx >= 1 and best_rss_norm / y_norm_sq < 1e-14:
            break

    if best_x6 is None or not np.all(np.isfinite(best_x6)):
        return np.full(6, np.nan, dtype=np.float64), float("nan"), False

    out = best_x6.copy()
    out[5] = out[5] * y_scale
    rss_unscaled = best_rss_norm * (y_scale * y_scale)
    return out, float(rss_unscaled), bool(best_ok or np.isfinite(rss_unscaled))


def fit_t2_ivim(
    signals: np.ndarray | Sequence[float],
    bvalues: np.ndarray | Sequence[float],
    te_values: np.ndarray | Sequence[float],
    bounds: Mapping[str, tuple[float, float]] | tuple[Sequence[float], Sequence[float]] | None = None,
    *,
    bounds_D: tuple[float, float] | None = None,
    bounds_D_star: tuple[float, float] | None = None,
    bounds_f: tuple[float, float] | None = None,
    bounds_T2_p: tuple[float, float] | None = None,
    bounds_T2_t: tuple[float, float] | None = None,
    b_thresh: float = DEFAULT_B_THRESH,
    rail_tol: float = DEFAULT_RAIL_TOL,
    grid: bool | None = None,
    mask: np.ndarray | None = None,
    max_nfev: int = 350,
    n_starts: int = 3,
) -> T2IVIMFitResult:
    """Fit the 2D ``(b, TE)`` relaxation-compensated IVIM model via bounded NLLS.

    Supports:
    - Single-voxel 1D paired ``(M,)`` or 2D cartesian ``(N_b, N_te)`` signal arrays,
    - Multi-voxel batched ``(N_vox, M)``, ``(..., M)``, or ``(..., N_b, N_te)`` arrays.

    Uses separable high-:math:`b` log-linear plane estimation plus deterministic
    multi-regime Golub–Pereyra variable-projection warm starts, followed by
    analytical-Jacobian Trust-Region Reflective bounded NLLS enforcing
    :math:`D \\le D^*`.

    Parameters
    ----------
    signals : np.ndarray or Sequence[float]
        Measured multi-echo DWI signals of shape ``(M,)``, ``(N_b, N_te)``,
        ``(N_vox, M)``, ``(..., M)``, or ``(..., N_b, N_te)``.
    bvalues : np.ndarray or Sequence[float]
        Diffusion :math:`b`-values in :math:`\\text{s}/\\text{mm}^2`.
    te_values : np.ndarray or Sequence[float]
        Echo times :math:`\\text{TE}` in :math:`\\text{ms}`.
    bounds : dict or 2-tuple, optional
        Parameter box bounds for ``("D", "D_star", "f", "T2_p", "T2_t")``.
    b_thresh : float, default=200.0
        High-:math:`b` threshold (:math:`\\text{s}/\\text{mm}^2`) for separable warm-start.
    rail_tol : float, default=1e-2
        Relative tolerance as a fraction of each parameter's bound span for
        boundary-railing classification.
    grid : bool or None, optional
        Explicitly treat 1D ``(bvalues, te_values)`` as a cartesian grid (``True``)
        or paired schedule (``False``).
    mask : np.ndarray, optional
        Boolean or binary spatial mask matching the leading voxel dimensions.
    max_nfev : int, default=350
        Maximum function evaluations per NLLS start.
    n_starts : int, default=3
        Number of ranked multi-regime warm starts refined per voxel.

    Returns
    -------
    T2IVIMFitResult
        Structured result with ``D``, ``D_star``, ``f``, ``T2_p``, ``T2_t``,
        ``S0``, ``railed_mask``, ``railed_params``, ``rss``, ``success``, and ``bounds``.
    """
    sig_arr = np.asarray(signals, dtype=np.float64)
    if sig_arr.ndim == 0:
        raise ValueError("`signals` must be at least 1D.")

    # Auto-detect equal-length small cartesian grids (e.g. 4 b x 4 TE -> (4, 4) signal)
    b_in = np.asarray(bvalues, dtype=np.float64)
    te_in = np.asarray(te_values, dtype=np.float64)
    effective_grid = grid
    if (
        effective_grid is None
        and b_in.ndim == 1
        and te_in.ndim == 1
        and b_in.size == te_in.size
        and sig_arr.ndim >= 2
        and sig_arr.shape[-2:] == (b_in.size, te_in.size)
        and b_in.size < 6
        and (b_in.size * te_in.size) >= 6
    ):
        effective_grid = True

    b_flat, te_flat, acq_shape, is_grid = _resolve_schedule(bvalues, te_values, grid=effective_grid)
    m_pts = int(b_flat.size)
    if m_pts < 6:
        raise ValueError(
            f"At least 6 (b, TE) measurements are required to fit 2D T2-IVIM; got {m_pts}."
        )
    if float(np.ptp(b_flat)) <= 0.0:
        raise ValueError("`bvalues` must contain at least 2 distinct values.")
    if float(np.ptp(te_flat)) <= 0.0:
        raise ValueError("`te_values` must contain at least 2 distinct echo times for 2D T2-IVIM.")

    bounds_dict = _normalize_bounds(
        bounds,
        bounds_D=bounds_D,
        bounds_D_star=bounds_D_star,
        bounds_f=bounds_f,
        bounds_T2_p=bounds_T2_p,
        bounds_T2_t=bounds_T2_t,
    )
    lo5 = np.array([bounds_dict[k][0] for k in PARAM_NAMES_5], dtype=np.float64)
    hi5 = np.array([bounds_dict[k][1] for k in PARAM_NAMES_5], dtype=np.float64)

    # Determine single-voxel vs multi-voxel spatial shape
    single_voxel = False
    if is_grid:
        nb, nte = acq_shape
        if sig_arr.shape == (nb, nte) or (sig_arr.ndim == 1 and sig_arr.size == m_pts):
            single_voxel = True
            spatial_shape: tuple[int, ...] = ()
            flat_sig = sig_arr.reshape(1, m_pts)
        elif sig_arr.ndim >= 3 and sig_arr.shape[-2:] == (nb, nte):
            spatial_shape = sig_arr.shape[:-2]
            flat_sig = sig_arr.reshape(int(np.prod(spatial_shape)), m_pts)
        elif sig_arr.ndim >= 2 and sig_arr.shape[-1] == m_pts:
            spatial_shape = sig_arr.shape[:-1]
            flat_sig = sig_arr.reshape(int(np.prod(spatial_shape)), m_pts)
        else:
            raise ValueError(
                f"`signals` shape {sig_arr.shape} is incompatible with cartesian grid "
                f"{acq_shape} (flattened length {m_pts})."
            )
    else:
        if sig_arr.ndim == 1 and sig_arr.size == m_pts:
            single_voxel = True
            spatial_shape = ()
            flat_sig = sig_arr.reshape(1, m_pts)
        elif sig_arr.ndim >= 2 and sig_arr.shape[-1] == m_pts:
            spatial_shape = sig_arr.shape[:-1]
            flat_sig = sig_arr.reshape(int(np.prod(spatial_shape)), m_pts)
        else:
            raise ValueError(
                f"`signals` shape {sig_arr.shape} must end with schedule length {m_pts}."
            )

    if single_voxel:
        if mask is not None and not bool(np.asarray(mask).item()):
            nan_v = float("nan")
            r_params_s: dict[str, np.ndarray | bool] = {k: False for k in PARAM_NAMES_5}
            return T2IVIMFitResult(
                D=nan_v,
                D_star=nan_v,
                f=nan_v,
                T2_p=nan_v,
                T2_t=nan_v,
                S0=nan_v,
                railed_mask=False,
                railed_params=r_params_s,
                rss=nan_v,
                success=False,
                bounds=bounds_dict,
                lower_mask=False,
                upper_mask=False,
            )

        est6, rss_val, ok_val = fit_t2_ivim_voxel(
            b_flat=b_flat,
            te_flat=te_flat,
            signal_flat=flat_sig[0],
            lo5=lo5,
            hi5=hi5,
            b_thresh=b_thresh,
            max_nfev=max_nfev,
            n_starts=n_starts,
        )
        railed_params_scalar: dict[str, np.ndarray | bool] = {}
        for idx_p, name in enumerate(PARAM_NAMES_5):
            r_info = classify_railing(
                float(est6[idx_p]), bounds_D_star=bounds_dict[name], rel_tol=rail_tol
            )
            railed_params_scalar[name] = bool(r_info["railed_mask"])
        ds_rail = classify_railing(
            float(est6[1]), bounds_D_star=bounds_dict["D_star"], rel_tol=rail_tol
        )
        return T2IVIMFitResult(
            D=float(est6[0]),
            D_star=float(est6[1]),
            f=float(est6[2]),
            T2_p=float(est6[3]),
            T2_t=float(est6[4]),
            S0=float(est6[5]),
            railed_mask=bool(ds_rail["railed_mask"]),
            railed_params=railed_params_scalar,
            rss=float(rss_val),
            success=bool(ok_val),
            bounds=bounds_dict,
            lower_mask=bool(ds_rail["lower_mask"]),
            upper_mask=bool(ds_rail["upper_mask"]),
        )

    n_vox = flat_sig.shape[0]
    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.shape != spatial_shape:
            raise ValueError(
                f"`mask` shape {mask_arr.shape} must match spatial shape {spatial_shape}."
            )
        active_mask = (mask_arr.reshape(n_vox) > 0) & np.all(np.isfinite(flat_sig), axis=1)
    else:
        active_mask = np.all(np.isfinite(flat_sig), axis=1) & (np.max(flat_sig, axis=1) > 0.0)

    params_flat = np.full((n_vox, 6), np.nan, dtype=np.float64)
    rss_flat = np.full(n_vox, np.nan, dtype=np.float64)
    success_flat = np.zeros(n_vox, dtype=bool)

    for idx in np.flatnonzero(active_mask):
        p6, r_v, ok_v = fit_t2_ivim_voxel(
            b_flat=b_flat,
            te_flat=te_flat,
            signal_flat=flat_sig[idx],
            lo5=lo5,
            hi5=hi5,
            b_thresh=b_thresh,
            max_nfev=max_nfev,
            n_starts=n_starts,
        )
        params_flat[idx] = p6
        rss_flat[idx] = r_v
        success_flat[idx] = ok_v

    d_map = params_flat[:, 0].reshape(spatial_shape)
    ds_map = params_flat[:, 1].reshape(spatial_shape)
    f_map = params_flat[:, 2].reshape(spatial_shape)
    t2p_map = params_flat[:, 3].reshape(spatial_shape)
    t2t_map = params_flat[:, 4].reshape(spatial_shape)
    s0_map = params_flat[:, 5].reshape(spatial_shape)
    rss_map = rss_flat.reshape(spatial_shape)
    ok_map = success_flat.reshape(spatial_shape)

    maps_by_name = {
        "D": d_map,
        "D_star": ds_map,
        "f": f_map,
        "T2_p": t2p_map,
        "T2_t": t2t_map,
    }
    railed_params_arr: dict[str, np.ndarray | bool] = {}
    for name in PARAM_NAMES_5:
        info = classify_railing(
            maps_by_name[name], bounds_D_star=bounds_dict[name], rel_tol=rail_tol
        )
        railed_params_arr[name] = np.asarray(info["railed_mask"], dtype=bool)

    ds_info = classify_railing(ds_map, bounds_D_star=bounds_dict["D_star"], rel_tol=rail_tol)
    return T2IVIMFitResult(
        D=d_map,
        D_star=ds_map,
        f=f_map,
        T2_p=t2p_map,
        T2_t=t2t_map,
        S0=s0_map,
        railed_mask=np.asarray(ds_info["railed_mask"], dtype=bool),
        railed_params=railed_params_arr,
        rss=rss_map,
        success=ok_map,
        bounds=bounds_dict,
        lower_mask=np.asarray(ds_info["lower_mask"], dtype=bool),
        upper_mask=np.asarray(ds_info["upper_mask"], dtype=bool),
    )


def compute_t2_ivim_jacobian(
    bvalues: np.ndarray | Sequence[float],
    te_values: np.ndarray | Sequence[float],
    D: float = 0.001,
    D_star: float = 0.025,
    f: float = 0.15,
    T2_p: float = 42.0,
    T2_t: float = 95.0,
    S0: float = 1.0,
    *,
    include_s0: bool = False,
    grid: bool | None = None,
) -> np.ndarray:
    """Compute the exact analytical Jacobian of :math:`S(b, \\text{TE})` wrt :math:`(D, D^*, f, T_{2,p}, T_{2,t})` (and optional :math:`S_0`).

    Parameters
    ----------
    bvalues : np.ndarray or Sequence[float]
        Diffusion :math:`b`-values (:math:`\\text{s}/\\text{mm}^2`).
    te_values : np.ndarray or Sequence[float]
        Echo times :math:`\\text{TE}` (:math:`\\text{ms}`).
    D, D_star, f, T2_p, T2_t, S0 : float
        Model parameter values at which to evaluate partial derivatives.
    include_s0 : bool, default=False
        If ``False``, returns shape ``(M, 5)`` for ``(D, D_star, f, T2_p, T2_t)``.
        If ``True``, appends ``dS/dS0`` as a 6th column of shape ``(M, 6)``.
    grid : bool or None, optional
        Cartesian grid override matching :func:`t2_ivim_forward_model`.

    Returns
    -------
    np.ndarray
        Analytical Jacobian matrix of shape ``(M, 5)`` (or ``(M, 6)``).
    """
    if float(T2_p) <= 0.0 or float(T2_t) <= 0.0:
        raise ValueError("`T2_p` and `T2_t` must be strictly positive (ms).")

    b_flat, te_flat, _, _ = _resolve_schedule(bvalues, te_values, grid=grid)
    d_v = float(D)
    ds_v = float(D_star)
    f_v = float(f)
    t2p_v = float(T2_p)
    t2t_v = float(T2_t)
    s0_v = float(S0)

    ep = np.exp(-b_flat * ds_v - te_flat / t2p_v)
    et = np.exp(-b_flat * d_v - te_flat / t2t_v)

    j_d = -b_flat * s0_v * (1.0 - f_v) * et
    j_ds = -b_flat * s0_v * f_v * ep
    j_f = s0_v * (ep - et)
    j_t2p = s0_v * f_v * ep * (te_flat / (t2p_v * t2p_v))
    j_t2t = s0_v * (1.0 - f_v) * et * (te_flat / (t2t_v * t2t_v))

    if include_s0:
        j_s0 = f_v * ep + (1.0 - f_v) * et
        return np.column_stack([j_d, j_ds, j_f, j_t2p, j_t2t, j_s0])
    return np.column_stack([j_d, j_ds, j_f, j_t2p, j_t2t])


def compute_t2_ivim_crlb(
    bvalues: np.ndarray | Sequence[float],
    te_values: np.ndarray | Sequence[float],
    D: float = 0.001,
    D_star: float = 0.025,
    f: float = 0.15,
    T2_p: float = 42.0,
    T2_t: float = 95.0,
    S0: float = 1.0,
    sigma: float | None = None,
    snr: float | None = None,
    *,
    include_s0: bool = False,
    reg: float = 0.0,
    grid: bool | None = None,
) -> dict[str, Any]:
    """Compute the exact analytical 5x5 Fisher Information Matrix and CRLB for 2D ``(b, TE)`` IVIM.

    By default (``include_s0=False``), evaluates the exact :math:`5 \\times 5` Fisher
    Information Matrix :math:`\\mathcal{I}(\\theta) = \\sigma^{-2} J^\\top J` over
    :math:`\\theta = (D, D^*, f, T_{2,p}, T_{2,t})`.

    Parameters
    ----------
    bvalues : np.ndarray or Sequence[float]
        Diffusion :math:`b`-values (:math:`\\text{s}/\\text{mm}^2`).
    te_values : np.ndarray or Sequence[float]
        Echo times :math:`\\text{TE}` (:math:`\\text{ms}`).
    D, D_star, f, T2_p, T2_t, S0 : float
        Ground-truth parameter values.
    sigma : float or None, optional
        Additive Gaussian thermal noise standard deviation :math:`\\sigma > 0`.
        Takes precedence if supplied; otherwise ``sigma = S0 / snr`` (default ``snr=50.0``).
    snr : float or None, optional
        Signal-to-noise ratio :math:`S_0 / \\sigma > 0` (used when ``sigma`` is ``None``;
        defaults to ``50.0`` if neither ``sigma`` nor ``snr`` is passed).
    include_s0 : bool, default=False
        If ``True``, includes nuisance :math:`S_0` as a 6th parameter (6x6 FIM).
    reg : float, default=0.0
        Optional diagonal Tikhonov regularization added before inversion.
    grid : bool or None, optional
        Cartesian grid override matching :func:`t2_ivim_forward_model`.

    Returns
    -------
    dict of str to Any
        Dictionary containing:
        - ``"fim"``: ``(5, 5)`` (or ``(6, 6)``) Fisher Information Matrix,
        - ``"cov"``: ``(5, 5)`` (or ``(6, 6)``) CRLB covariance matrix,
        - ``"crlb_std"``: dict mapping each parameter name to its CRLB standard deviation,
        - ``"std"``: 1D ``np.ndarray`` of CRLB standard deviations in parameter order,
        - ``"condition_number"``: condition number ``np.linalg.cond(fim)``,
        - ``"jacobian"``: analytical Jacobian matrix ``J``,
        - ``"cv"``: dict of parameter CRLB coefficients of variation,
        - ``"cv_D_star"``: CRLB coefficient of variation for :math:`D^*`,
        - ``"sigma"``: effective noise standard deviation :math:`\\sigma`,
        - ``"param_names"``: tuple of parameter names.
    """
    if sigma is not None:
        sigma_val = float(sigma)
        if sigma_val <= 0.0 or not np.isfinite(sigma_val):
            raise ValueError(f"`sigma` must be a positive finite number; got {sigma!r}.")
    else:
        snr_val = 50.0 if snr is None else float(snr)
        if snr_val <= 0.0 or not np.isfinite(snr_val):
            raise ValueError(f"`snr` must be a positive finite number; got {snr!r}.")
        sigma_val = float(S0) / snr_val

    jac = compute_t2_ivim_jacobian(
        bvalues=bvalues,
        te_values=te_values,
        D=D,
        D_star=D_star,
        f=f,
        T2_p=T2_p,
        T2_t=T2_t,
        S0=S0,
        include_s0=include_s0,
        grid=grid,
    )
    n_p = jac.shape[1]
    fim = (jac.T @ jac) / (sigma_val * sigma_val)
    cond = float(np.linalg.cond(fim))

    fim_reg = fim + float(reg) * np.eye(n_p, dtype=np.float64) if reg > 0.0 else fim
    try:
        cov = np.linalg.inv(fim_reg)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fim_reg)

    std_arr = np.sqrt(np.maximum(np.diag(cov), 0.0))
    names = PARAM_NAMES_6 if include_s0 else PARAM_NAMES_5
    vals = (D, D_star, f, T2_p, T2_t, S0) if include_s0 else (D, D_star, f, T2_p, T2_t)

    crlb_std_dict = {name: float(std_arr[i]) for i, name in enumerate(names)}
    cv_dict = {
        name: float(std_arr[i] / max(abs(float(vals[i])), 1e-12))
        for i, name in enumerate(names)
    }

    return {
        "fim": fim,
        "cov": cov,
        "crlb_std": crlb_std_dict,
        "std": std_arr,
        "condition_number": cond,
        "jacobian": jac,
        "cv": cv_dict,
        "cv_D_star": cv_dict["D_star"],
        "sigma": sigma_val,
        "param_names": names,
    }


@dataclass
class RailingComparisonResult(Mapping[str, Any]):
    """Monte Carlo & CRLB comparison between 1D single-TE and 2D ``(b, TE)`` IVIM."""

    railing_rate_1d: float
    railing_rate_2d: float
    railing_reduction: float
    railing_relative_reduction: float
    f_bias_1d: float
    f_bias_2d: float
    f_rmse_1d: float
    f_rmse_2d: float
    f_apparent_1d: float
    d_star_rmse_1d: float
    d_star_rmse_2d: float
    d_rmse_1d: float
    d_rmse_2d: float
    crlb_1d: dict[str, Any]
    crlb_2d: dict[str, Any]
    crlb_d_star_std_1d: float
    crlb_d_star_std_2d: float
    n_trials: int
    snr: float
    te_1d: float
    match_scan_time: bool
    seed: int | None

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> list[str]:  # type: ignore[override]
        return [
            "railing_rate_1d",
            "railing_rate_2d",
            "railing_reduction",
            "railing_relative_reduction",
            "f_bias_1d",
            "f_bias_2d",
            "f_rmse_1d",
            "f_rmse_2d",
            "f_apparent_1d",
            "d_star_rmse_1d",
            "d_star_rmse_2d",
            "d_rmse_1d",
            "d_rmse_2d",
            "crlb_1d",
            "crlb_2d",
            "crlb_d_star_std_1d",
            "crlb_d_star_std_2d",
            "n_trials",
            "snr",
            "te_1d",
            "match_scan_time",
            "seed",
        ]

    def __getitem__(self, key: str) -> Any:
        data = self.to_dict()
        if key in data:
            return data[key]
        raise KeyError(f"Unknown key {key!r} in RailingComparisonResult")

    def to_dict(self) -> dict[str, Any]:
        """Serialize comparison diagnostics to a dictionary."""
        return {
            "railing_rate_1d": self.railing_rate_1d,
            "railing_rate_2d": self.railing_rate_2d,
            "railing_reduction": self.railing_reduction,
            "railing_relative_reduction": self.railing_relative_reduction,
            "f_bias_1d": self.f_bias_1d,
            "f_bias_2d": self.f_bias_2d,
            "f_rmse_1d": self.f_rmse_1d,
            "f_rmse_2d": self.f_rmse_2d,
            "f_apparent_1d": self.f_apparent_1d,
            "d_star_rmse_1d": self.d_star_rmse_1d,
            "d_star_rmse_2d": self.d_star_rmse_2d,
            "d_rmse_1d": self.d_rmse_1d,
            "d_rmse_2d": self.d_rmse_2d,
            "crlb_1d": self.crlb_1d,
            "crlb_2d": self.crlb_2d,
            "crlb_d_star_std_1d": self.crlb_d_star_std_1d,
            "crlb_d_star_std_2d": self.crlb_d_star_std_2d,
            "n_trials": self.n_trials,
            "snr": self.snr,
            "te_1d": self.te_1d,
            "match_scan_time": self.match_scan_time,
            "seed": self.seed,
        }


def compare_1d_vs_2d_railing(
    bvalues: np.ndarray | Sequence[float] | None = None,
    te_values: np.ndarray | Sequence[float] | None = None,
    *,
    te_1d: float = 85.0,
    D: float = 0.001,
    D_star: float = 0.025,
    f: float = 0.16,
    T2_p: float = 45.0,
    T2_t: float = 95.0,
    S0: float = 1.0,
    snr: float = 65.0,
    n_trials: int = 128,
    noise_type: Literal["gaussian", "rician"] = "gaussian",
    method_1d: Literal["nlls", "segmented"] = "nlls",
    match_scan_time: bool = True,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
    seed: int | np.random.Generator | None = GLOBAL_SEED,
) -> RailingComparisonResult:
    """Compare 1D single-TE ``S(b)`` vs 2D multi-echo ``S(b, TE)`` boundary-railing and bias.

    Under realistic microvascular versus parenchymal :math:`T_2` relaxation
    contrast (default: venous/capillary blood :math:`T_{2,p} = 45\\,\\text{ms}` vs
    parenchymal tissue :math:`T_{2,t} = 95\\,\\text{ms}` at clinical single
    :math:`\\text{TE}_{1\\text{D}} = 85\\,\\text{ms}`), a single-TE 1D acquisition
    suffers differential :math:`T_2` attenuation of the microvascular pool,
    inducing systematic echo-time bias in :math:`\\hat{f}` and elevated :math:`D^*`
    boundary-railing even when matched for total scan time (number of shots).
    Joint 2D ``(b, TE)`` fitting captures short-TE microvascular signal and
    disentangles :math:`(T_{2,p}, T_{2,t})`, reducing both :math:`D^*` railing
    and :math:`f` echo-time bias.

    Parameters
    ----------
    bvalues : np.ndarray or Sequence[float], optional
        1D array of diffusion :math:`b`-values (:math:`\\text{s}/\\text{mm}^2`).
        Defaults to ``[0, 15, 30, 60, 120, 250, 500, 800]``.
    te_values : np.ndarray or Sequence[float], optional
        1D array of echo times (:math:`\\text{ms}`) for 2D ``(b, TE)`` acquisition.
        Defaults to ``[22, 42, 70, 110]``.
    te_1d : float, default=85.0
        Single clinical echo time (:math:`\\text{ms}`) used for the 1D acquisition.
    D, D_star, f, T2_p, T2_t, S0 : float
        Ground-truth two-compartment diffusion-relaxation parameters.
    snr : float, default=65.0
        Thermal baseline SNR defined at :math:`(b = 0, \\text{TE} = 0)` so per-shot
        noise standard deviation is :math:`\\sigma = S_0 / \\text{snr}`.
    n_trials : int, default=128
        Number of Monte Carlo realizations (seeded deterministically via ``PCG64``).
    noise_type : {"gaussian", "rician"}, default="gaussian"
        Noise model applied to each simulated acquisition shot.
    method_1d : {"nlls", "segmented"}, default="nlls"
        1D IVIM solver passed to :func:`~sextant_core.fitting.fit_ivim`.
    match_scan_time : bool, default=True
        If ``True``, gives the 1D single-TE acquisition ``len(te_values)`` signal
        averages (NEX) per :math:`b`-value so both 1D and 2D protocols use the
        exact same total number of EPI shots.
    bounds : dict, optional
        Shared parameter bounds for ``D``, ``D_star``, ``f``, ``T2_p``, ``T2_t``.
    rail_tol : float, default=1e-2
        Relative boundary-railing tolerance.
    seed : int, np.random.Generator, or None, default=20260613
        Deterministic seed initialized via ``np.random.Generator(np.random.PCG64(seed))``.

    Returns
    -------
    RailingComparisonResult
        Mapping-compatible dataclass containing 1D vs 2D railing rates,
        :math:`f` echo-time bias, :math:`D^*` RMSE, and CRLB comparisons.
    """
    if n_trials < 1:
        raise ValueError(f"`n_trials` must be >= 1; got {n_trials}.")
    if float(snr) <= 0.0 or not np.isfinite(snr):
        raise ValueError(f"`snr` must be positive and finite; got {snr!r}.")
    if noise_type not in ("gaussian", "rician"):
        raise ValueError(f"Unsupported `noise_type` {noise_type!r}.")

    b_arr = (
        DEFAULT_T2_IVIM_BVALS.copy()
        if bvalues is None
        else np.asarray(bvalues, dtype=np.float64)
    )
    te_arr = (
        DEFAULT_T2_IVIM_TES.copy()
        if te_values is None
        else np.asarray(te_values, dtype=np.float64)
    )
    if b_arr.ndim != 1 or te_arr.ndim != 1:
        raise ValueError("`bvalues` and `te_values` for `compare_1d_vs_2d_railing` must be 1D.")

    bounds_dict = _normalize_bounds(bounds)
    rng = _make_pcg64_rng(seed)
    seed_int = seed if isinstance(seed, int) else (GLOBAL_SEED if seed is None else None)

    sigma_shot = float(S0) / float(snr)
    nex_1d = int(te_arr.size) if match_scan_time else 1

    # Clean 2D cartesian signal of shape (N_b, N_te)
    clean_2d = t2_ivim_forward_model(
        b_arr, te_arr, D=D, D_star=D_star, f=f, T2_p=T2_p, T2_t=T2_t, S0=S0, grid=True
    )
    # Clean 1D signal at single echo time te_1d of shape (N_b,)
    clean_1d = t2_ivim_forward_model(
        b_arr,
        np.full_like(b_arr, float(te_1d)),
        D=D,
        D_star=D_star,
        f=f,
        T2_p=T2_p,
        T2_t=T2_t,
        S0=S0,
        grid=False,
    )

    # Simulate 2D trials: (n_trials, N_b, N_te)
    if noise_type == "rician":
        re_2d = clean_2d[None, :, :] + rng.normal(
            0.0, sigma_shot, size=(n_trials, b_arr.size, te_arr.size)
        )
        im_2d = rng.normal(0.0, sigma_shot, size=(n_trials, b_arr.size, te_arr.size))
        noisy_2d = np.sqrt(re_2d * re_2d + im_2d * im_2d)

        re_1d = clean_1d[None, None, :] + rng.normal(
            0.0, sigma_shot, size=(n_trials, nex_1d, b_arr.size)
        )
        im_1d = rng.normal(0.0, sigma_shot, size=(n_trials, nex_1d, b_arr.size))
        noisy_1d = np.mean(np.sqrt(re_1d * re_1d + im_1d * im_1d), axis=1)
    else:
        eps_2d = rng.normal(0.0, sigma_shot, size=(n_trials, b_arr.size, te_arr.size))
        noisy_2d = np.clip(clean_2d[None, :, :] + eps_2d, 1e-12, None)

        eps_1d = rng.normal(0.0, sigma_shot, size=(n_trials, nex_1d, b_arr.size))
        noisy_1d = np.clip(np.mean(clean_1d[None, None, :] + eps_1d, axis=1), 1e-12, None)

    fit_1d = fit_ivim(
        b_arr,
        noisy_1d,
        method=method_1d,
        bounds_D=bounds_dict["D"],
        bounds_D_star=bounds_dict["D_star"],
        bounds_f=bounds_dict["f"],
        rail_tol=rail_tol,
    )
    fit_2d = fit_t2_ivim(
        noisy_2d,
        b_arr,
        te_arr,
        bounds=bounds_dict,
        rail_tol=rail_tol,
        grid=True,
    )

    rail_1d = float(fit_1d.frac_railed)
    rail_2d = float(fit_2d.frac_railed)

    f_1d = np.asarray(fit_1d.f, dtype=np.float64)
    f_2d = np.asarray(fit_2d.f, dtype=np.float64)
    ds_1d = np.asarray(fit_1d.D_star, dtype=np.float64)
    ds_2d = np.asarray(fit_2d.D_star, dtype=np.float64)
    d_1d = np.asarray(fit_1d.D, dtype=np.float64)
    d_2d = np.asarray(fit_2d.D, dtype=np.float64)

    f_bias_1d = float(np.nanmean(f_1d - f))
    f_bias_2d = float(np.nanmean(f_2d - f))
    f_rmse_1d = float(np.sqrt(np.nanmean((f_1d - f) ** 2)))
    f_rmse_2d = float(np.sqrt(np.nanmean((f_2d - f) ** 2)))
    ds_rmse_1d = float(np.sqrt(np.nanmean((ds_1d - D_star) ** 2)))
    ds_rmse_2d = float(np.sqrt(np.nanmean((ds_2d - D_star) ** 2)))
    d_rmse_1d = float(np.sqrt(np.nanmean((d_1d - D) ** 2)))
    d_rmse_2d = float(np.sqrt(np.nanmean((d_2d - D) ** 2)))

    # Theoretical apparent f and effective SNR at single echo time te_1d
    w_p = float(f) * float(np.exp(-float(te_1d) / float(T2_p)))
    w_t = (1.0 - float(f)) * float(np.exp(-float(te_1d) / float(T2_t)))
    s0_eff_1d = float(S0) * (w_p + w_t)
    f_app_1d = w_p / max(w_p + w_t, 1e-30)
    sigma_eff_1d = sigma_shot / float(np.sqrt(nex_1d))
    snr_eff_1d = s0_eff_1d / sigma_eff_1d

    crlb_1d = compute_crlb(
        b_arr, D=D, D_star=D_star, f=f_app_1d, snr=snr_eff_1d, S0=s0_eff_1d, reg=0.0
    )
    crlb_2d = compute_t2_ivim_crlb(
        b_arr,
        te_arr,
        D=D,
        D_star=D_star,
        f=f,
        T2_p=T2_p,
        T2_t=T2_t,
        S0=S0,
        sigma=sigma_shot,
        grid=True,
    )

    crlb_ds_1d = float(np.asarray(crlb_1d["std"])[1])
    crlb_ds_2d = float(crlb_2d["crlb_std"]["D_star"])
    rail_red = rail_1d - rail_2d
    rail_rel_red = rail_red / max(rail_1d, 1e-12) if rail_1d > 0.0 else 0.0

    return RailingComparisonResult(
        railing_rate_1d=rail_1d,
        railing_rate_2d=rail_2d,
        railing_reduction=rail_red,
        railing_relative_reduction=rail_rel_red,
        f_bias_1d=f_bias_1d,
        f_bias_2d=f_bias_2d,
        f_rmse_1d=f_rmse_1d,
        f_rmse_2d=f_rmse_2d,
        f_apparent_1d=f_app_1d,
        d_star_rmse_1d=ds_rmse_1d,
        d_star_rmse_2d=ds_rmse_2d,
        d_rmse_1d=d_rmse_1d,
        d_rmse_2d=d_rmse_2d,
        crlb_1d=crlb_1d,
        crlb_2d=crlb_2d,
        crlb_d_star_std_1d=crlb_ds_1d,
        crlb_d_star_std_2d=crlb_ds_2d,
        n_trials=int(n_trials),
        snr=float(snr),
        te_1d=float(te_1d),
        match_scan_time=bool(match_scan_time),
        seed=seed_int,
    )
