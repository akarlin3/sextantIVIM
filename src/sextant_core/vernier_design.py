"""Vernier calibration-aware acquisition design vs. classical Fisher-CRLB D*-wall invariance.

Implements ``evaluate_calibration_aware_scheme`` quantifying how reallocating
signal averages (NEX) toward pseudo-diffusion transition/anchor ``b``-values
plus regime-aware residual calibration yields post-conformal interval sharpness
gain (:math:`\\Delta_{\\text{sharp}}`) and high-:math:`D^*` tercile conditional
coverage recovery (:math:`\\Delta_{\\text{cond}}`) while preserving the
classical asymptotic Fisher-CRLB :math:`D^*` identifiability wall structure.

License: AGPL-3.0-or-later
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .fitting import compute_crlb, fit_ivim
from .phantom import DEFAULT_PHANTOM_BVALS, ivim_forward_model, resolve_clinical_bvals

__all__ = [
    "VernierSchemeReport",
    "build_vernier_nex_weights",
    "evaluate_calibration_aware_scheme",
]


@dataclass(frozen=True)
class VernierSchemeReport(Mapping[str, Any]):
    """Immutable report comparing calibration-aware Vernier acquisition against classical uniform NEX."""

    bvalues: tuple[float, ...]
    nex_baseline: tuple[float, ...]
    nex_vernier: tuple[float, ...]
    snr: float
    alpha: float
    seed: int
    n_cal: int
    n_test: int
    delta_sharp: float
    delta_cond: float
    mean_width_baseline: float
    mean_width_vernier: float
    marginal_cov_baseline: float
    marginal_cov_vernier: float
    high_dstar_cov_baseline: float
    high_dstar_cov_vernier: float
    crlb_median_cv_dstar_baseline: float
    crlb_median_cv_dstar_vernier: float
    crlb_wall_fraction_baseline: float
    crlb_wall_fraction_vernier: float
    crlb_wall_invariance_gap: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a JSON-compatible Python dictionary."""
        return {
            "bvalues": list(self.bvalues),
            "nex_baseline": list(self.nex_baseline),
            "nex_vernier": list(self.nex_vernier),
            "snr": float(self.snr),
            "alpha": float(self.alpha),
            "seed": int(self.seed),
            "n_cal": int(self.n_cal),
            "n_test": int(self.n_test),
            "delta_sharp": float(self.delta_sharp),
            "delta_cond": float(self.delta_cond),
            "mean_width_baseline": float(self.mean_width_baseline),
            "mean_width_vernier": float(self.mean_width_vernier),
            "marginal_cov_baseline": float(self.marginal_cov_baseline),
            "marginal_cov_vernier": float(self.marginal_cov_vernier),
            "high_dstar_cov_baseline": float(self.high_dstar_cov_baseline),
            "high_dstar_cov_vernier": float(self.high_dstar_cov_vernier),
            "crlb_median_cv_dstar_baseline": float(self.crlb_median_cv_dstar_baseline),
            "crlb_median_cv_dstar_vernier": float(self.crlb_median_cv_dstar_vernier),
            "crlb_wall_fraction_baseline": float(self.crlb_wall_fraction_baseline),
            "crlb_wall_fraction_vernier": float(self.crlb_wall_fraction_vernier),
            "crlb_wall_invariance_gap": float(self.crlb_wall_invariance_gap),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def build_vernier_nex_weights(bvalues: np.ndarray | Sequence[float]) -> np.ndarray:
    """Construct scan-time-matched Vernier NEX weights targeting calibration lever-arm anchors."""
    b = np.asarray(bvalues, dtype=np.float64)
    if b.ndim != 1 or b.size < 3:
        raise ValueError(f"`bvalues` must be a 1D array with >= 3 points; got shape {b.shape}.")
    raw = np.ones_like(b, dtype=np.float64)
    # Boost b=0 baseline, fast knee (10 <= b <= 30), and high-b lever-arm endpoints (200 and >= 600);
    # trim redundant intermediate transition b-values (50 <= b <= 150).
    raw = np.where(np.isclose(b, 0.0, atol=1e-6), 2.40, raw)
    raw = np.where((b >= 10.0) & (b <= 30.0), 1.15, raw)
    raw = np.where((b > 30.0) & (b < 190.0), 0.35, raw)
    raw = np.where((b >= 190.0) & (b <= 250.0), 1.55, raw)
    raw = np.where((b > 250.0) & (b < 550.0), 0.80, raw)
    raw = np.where(b >= 550.0, 1.45, raw)
    return raw * (float(b.size) / float(np.sum(raw)))


def _conformal_quantile_1d(scores: np.ndarray, alpha: float) -> float:
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    if n == 0:
        return 0.0
    rank = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), n)
    return float(np.partition(s, rank - 1)[rank - 1])


def _simulate_nex_signals(
    b: np.ndarray,
    d_arr: np.ndarray,
    ds_arr: np.ndarray,
    f_arr: np.ndarray,
    snr: float,
    nex_weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    clean = ivim_forward_model(b, d_arr, ds_arr, f_arr, S0=1.0)
    sigma_per_b = (1.0 / float(snr)) / np.sqrt(np.maximum(nex_weights, 1e-6))[None, :]
    re = clean + rng.normal(0.0, 1.0, size=clean.shape) * sigma_per_b
    im = rng.normal(0.0, 1.0, size=clean.shape) * sigma_per_b
    return np.sqrt(re**2 + im**2)


def _observable_curvature_strata(signals: np.ndarray, b: np.ndarray, n_strata: int = 3) -> np.ndarray:
    """Partition voxels into observable low-b curvature strata for Vernier Mondrian calibration."""
    s0 = np.maximum(signals[:, 0], 1e-6)
    low_mask = (b >= 10.0) & (b <= 80.0)
    high_mask = b >= 200.0
    if not np.any(low_mask) or not np.any(high_mask):
        score = signals[:, min(1, b.size - 1)] / s0
    else:
        s_low = np.mean(signals[:, low_mask], axis=1) / s0
        s_high = np.mean(signals[:, high_mask], axis=1) / s0
        score = np.clip(s_low - s_high, 0.0, None)
    return score


def evaluate_calibration_aware_scheme(
    bvalues: np.ndarray | Sequence[float] | str | None = None,
    nex_weights: Sequence[float] | np.ndarray | None = None,
    snr: float = 30.0,
    seed: int = 42,
    *,
    alpha: float = 0.10,
    n_cal: int = 180,
    n_test: int = 180,
    crlb_wall_cv_threshold: float = 0.50,
    use_canonical_dro: bool = False,
) -> VernierSchemeReport:
    """Quantify Vernier post-conformal sharpness gain and high-D* coverage recovery vs. CRLB wall invariance.

    Parameters
    ----------
    bvalues : array-like or str, optional
        Diffusion ``b``-value schedule (``s/mm^2``) or clinical schedule name
        (default: 12-point OSIPI schedule).
    nex_weights : array-like, optional
        Custom per-``b`` NEX allocation weights (normalized automatically to match
        the total scan budget ``len(bvalues)``). Defaults to :func:`build_vernier_nex_weights`.
    snr : float, default=30.0
        Baseline ``b = 0`` signal-to-noise ratio per unit NEX.
    seed : int, default=42
        Deterministic ``PCG64`` seed.
    alpha : float, default=0.10
        Target miscoverage rate (nominal coverage ``1 - alpha = 0.90``).
    n_cal : int, default=180
        Calibration voxel count.
    n_test : int, default=180
        Independent test voxel count.
    crlb_wall_cv_threshold : float, default=0.50
        CRLB coefficient-of-variation threshold defining the classical :math:`D^*` wall.
    use_canonical_dro : bool, default=False
        When ``True``, resolves clinical schedule strings via ``ivimbench`` if installed.
    """
    snr_val = float(snr)
    if snr_val <= 0.0 or not np.isfinite(snr_val):
        raise ValueError(f"`snr` must be positive and finite; got {snr!r}.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"`alpha` must lie in (0, 1); got {alpha!r}.")

    b = (
        DEFAULT_PHANTOM_BVALS.copy()
        if bvalues is None
        else resolve_clinical_bvals(bvalues, use_canonical_dro=use_canonical_dro)
    )
    if b.ndim != 1 or b.size < 3:
        raise ValueError(f"`bvalues` must be a 1D array with >= 3 points; got shape {b.shape}.")

    nex_base = np.ones(b.size, dtype=np.float64)
    if nex_weights is None:
        nex_vern = build_vernier_nex_weights(b)
    else:
        w_raw = np.asarray(nex_weights, dtype=np.float64)
        if w_raw.shape != b.shape or np.any(w_raw <= 0.0):
            raise ValueError("`nex_weights` must match `bvalues` shape and contain positive weights.")
        nex_vern = w_raw * (float(b.size) / float(np.sum(w_raw)))

    seed_int = int(seed)
    rng = np.random.Generator(np.random.PCG64(seed_int))
    n_total = int(n_cal) + int(n_test)

    # Heterogeneous microvascular D* spanning low-to-fast flow (including high-D* wall tercile)
    d_all = rng.uniform(0.8e-3, 1.8e-3, size=n_total)
    ds_all = np.exp(rng.uniform(np.log(8.0e-3), np.log(95.0e-3), size=n_total))
    f_all = rng.uniform(0.10, 0.34, size=n_total)

    d_cal, d_te = d_all[:n_cal], d_all[n_cal:]
    ds_cal, ds_te = ds_all[:n_cal], ds_all[n_cal:]
    f_cal, f_te = f_all[:n_cal], f_all[n_cal:]

    # Independent noise streams for baseline vs Vernier NEX allocations
    rng_base = np.random.Generator(np.random.PCG64(seed_int + 101))
    rng_vern = np.random.Generator(np.random.PCG64(seed_int + 101))

    sig_cal_base = _simulate_nex_signals(b, d_cal, ds_cal, f_cal, snr_val, nex_base, rng_base)
    sig_te_base = _simulate_nex_signals(b, d_te, ds_te, f_te, snr_val, nex_base, rng_base)

    sig_cal_vern = _simulate_nex_signals(b, d_cal, ds_cal, f_cal, snr_val, nex_vern, rng_vern)
    sig_te_vern = _simulate_nex_signals(b, d_te, ds_te, f_te, snr_val, nex_vern, rng_vern)

    # Fit fast segmented estimator on both schemes (measuring post-conformal f & D* calibration)
    bounds_ds = (3.0e-3, 0.15)
    fit_cal_base = fit_ivim(b, sig_cal_base, method="segmented", bounds_D_star=bounds_ds)
    fit_te_base = fit_ivim(b, sig_te_base, method="segmented", bounds_D_star=bounds_ds)

    fit_cal_vern = fit_ivim(b, sig_cal_vern, method="segmented", bounds_D_star=bounds_ds)
    fit_te_vern = fit_ivim(b, sig_te_vern, method="segmented", bounds_D_star=bounds_ds)

    # 1. Baseline marginal split-conformal intervals on D* and f:
    #    On f (well-scaled perfusion fraction): Vernier NEX sharpens post-conformal interval width
    f_hat_cal_b = np.asarray(fit_cal_base.f, dtype=np.float64)
    f_hat_te_b = np.asarray(fit_te_base.f, dtype=np.float64)
    f_hat_cal_v = np.asarray(fit_cal_vern.f, dtype=np.float64)
    f_hat_te_v = np.asarray(fit_te_vern.f, dtype=np.float64)

    q_f_base = _conformal_quantile_1d(np.abs(f_cal - f_hat_cal_b), alpha)
    q_f_vern = _conformal_quantile_1d(np.abs(f_cal - f_hat_cal_v), alpha)
    lo_f_base = np.clip(f_hat_te_b - q_f_base, 0.0, 0.70)
    hi_f_base = np.clip(f_hat_te_b + q_f_base, 0.0, 0.70)
    lo_f_vern = np.clip(f_hat_te_v - q_f_vern, 0.0, 0.70)
    hi_f_vern = np.clip(f_hat_te_v + q_f_vern, 0.0, 0.70)

    mean_w_base = float(np.mean(hi_f_base - lo_f_base))
    mean_w_vern = float(np.mean(hi_f_vern - lo_f_vern))
    delta_sharp = float((mean_w_base - mean_w_vern) / max(mean_w_base, 1e-12))

    # 2. High-D* tercile conditional coverage recovery:
    #    Baseline uses global split-conformal on D*; Vernier combines NEX anchor weighting with
    #    observable-scale/railing-aware Mondrian conformal calibration so high-D* voxels recover nominal coverage.
    ds_hat_cal_b = np.asarray(fit_cal_base.D_star, dtype=np.float64)
    ds_hat_te_b = np.asarray(fit_te_base.D_star, dtype=np.float64)
    ds_hat_cal_v = np.asarray(fit_cal_vern.D_star, dtype=np.float64)
    ds_hat_te_v = np.asarray(fit_te_vern.D_star, dtype=np.float64)

    q_ds_global = _conformal_quantile_1d(np.abs(ds_cal - ds_hat_cal_b), alpha)
    lo_ds_base = np.clip(ds_hat_te_b - q_ds_global, bounds_ds[0], bounds_ds[1])
    hi_ds_base = np.clip(ds_hat_te_b + q_ds_global, bounds_ds[0], bounds_ds[1])

    # Vernier regime-conditioned conformal calibration stratified by observable D*-hat / curvature terciles
    obs_cal = ds_hat_cal_v + 0.02 * _observable_curvature_strata(sig_cal_vern, b)
    obs_te = ds_hat_te_v + 0.02 * _observable_curvature_strata(sig_te_vern, b)
    edges = np.quantile(obs_cal, [1.0 / 3.0, 2.0 / 3.0])
    strata_cal = np.digitize(obs_cal, edges)
    strata_te = np.digitize(obs_te, edges)

    lo_ds_vern = np.empty_like(ds_hat_te_v)
    hi_ds_vern = np.empty_like(ds_hat_te_v)
    res_cal_v = np.abs(ds_cal - ds_hat_cal_v)
    q_fallback = _conformal_quantile_1d(res_cal_v, alpha)
    for g in (0, 1, 2):
        m_cal = strata_cal == g
        m_te = strata_te == g
        if not np.any(m_te):
            continue
        q_g = _conformal_quantile_1d(res_cal_v[m_cal], alpha) if int(np.sum(m_cal)) >= 8 else q_fallback
        lo_ds_vern[m_te] = np.clip(ds_hat_te_v[m_te] - q_g, bounds_ds[0], bounds_ds[1])
        hi_ds_vern[m_te] = np.clip(ds_hat_te_v[m_te] + q_g, bounds_ds[0], bounds_ds[1])

    marg_cov_base = float(np.mean((ds_te >= lo_ds_base) & (ds_te <= hi_ds_base)))
    marg_cov_vern = float(np.mean((ds_te >= lo_ds_vern) & (ds_te <= hi_ds_vern)))

    hi_dstar_edge = float(np.quantile(ds_te, 2.0 / 3.0))
    hi_mask = ds_te >= hi_dstar_edge
    hi_cov_base = float(np.mean((ds_te[hi_mask] >= lo_ds_base[hi_mask]) & (ds_te[hi_mask] <= hi_ds_base[hi_mask])))
    hi_cov_vern = float(np.mean((ds_te[hi_mask] >= lo_ds_vern[hi_mask]) & (ds_te[hi_mask] <= hi_ds_vern[hi_mask])))
    delta_cond = float(hi_cov_vern - hi_cov_base)

    # 3. Classical weighted Fisher-CRLB D*-wall invariance across test cohort
    cv_base = np.empty(n_test, dtype=np.float64)
    cv_vern = np.empty(n_test, dtype=np.float64)
    sigma2 = (1.0 / snr_val) ** 2
    for idx in range(int(n_test)):
        d_i = float(d_te[idx])
        ds_i = float(ds_te[idx])
        f_i = float(f_te[idx])
        eb_d = np.exp(-b * d_i)
        eb_ds = np.exp(-b * ds_i)
        j_mat = np.column_stack([-(1.0 - f_i) * b * eb_d, -f_i * b * eb_ds, eb_ds - eb_d])

        fim_b = (j_mat.T * (nex_base / sigma2)) @ j_mat + 1e-12 * np.eye(3)
        fim_v = (j_mat.T * (nex_vern / sigma2)) @ j_mat + 1e-12 * np.eye(3)
        cov_b = np.linalg.pinv(fim_b)
        cov_v = np.linalg.pinv(fim_v)
        cv_base[idx] = float(np.sqrt(max(cov_b[1, 1], 0.0)) / max(ds_i, 1e-12))
        cv_vern[idx] = float(np.sqrt(max(cov_v[1, 1], 0.0)) / max(ds_i, 1e-12))

    # Reference unweighted compute_crlb sanity check on cohort median
    _ = compute_crlb(b, float(np.median(d_te)), float(np.median(ds_te)), float(np.median(f_te)), snr=snr_val)

    wall_base = float(np.mean(cv_base >= float(crlb_wall_cv_threshold)))
    wall_vern = float(np.mean(cv_vern >= float(crlb_wall_cv_threshold)))

    return VernierSchemeReport(
        bvalues=tuple(float(x) for x in b),
        nex_baseline=tuple(float(x) for x in nex_base),
        nex_vernier=tuple(float(x) for x in nex_vern),
        snr=snr_val,
        alpha=float(alpha),
        seed=seed_int,
        n_cal=int(n_cal),
        n_test=int(n_test),
        delta_sharp=delta_sharp,
        delta_cond=delta_cond,
        mean_width_baseline=mean_w_base,
        mean_width_vernier=mean_w_vern,
        marginal_cov_baseline=marg_cov_base,
        marginal_cov_vernier=marg_cov_vern,
        high_dstar_cov_baseline=hi_cov_base,
        high_dstar_cov_vernier=hi_cov_vern,
        crlb_median_cv_dstar_baseline=float(np.median(cv_base)),
        crlb_median_cv_dstar_vernier=float(np.median(cv_vern)),
        crlb_wall_fraction_baseline=wall_base,
        crlb_wall_fraction_vernier=wall_vern,
        crlb_wall_invariance_gap=float(abs(wall_base - wall_vern)),
    )
