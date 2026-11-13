"""Procrustes model-misspecification sensitivity & conditional coverage aliasing probe.

Implements ``probe_d_misspecification_aliasing`` quantifying how fitting a
canonical bi-exponential IVIM model + global split-conformal intervals to
non-bi-exponential microvascular dispersion / heavy-tail truth preserves global
marginal :math:`1 - \\alpha` coverage on the well-identified parenchymal
diffusivity :math:`D` while inducing systematic conditional coverage aliasing
across perfusion/dispersion strata.

License: AGPL-3.0-or-later
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .fitting import fit_ivim
from .phantom import (
    DEFAULT_PHANTOM_BVALS,
    generate_synthetic_phantom,
    resolve_clinical_bvals,
)

__all__ = [
    "ProcrustesProbeReport",
    "probe_d_misspecification_aliasing",
]


@dataclass(frozen=True)
class ProcrustesProbeReport(Mapping[str, Any]):
    """Immutable report of global vs. conditional D coverage under bi-exponential misspecification."""

    family: str
    bvalues: tuple[float, ...]
    snr: float
    alpha: float
    nominal_coverage: float
    seed: int
    n_cal: int
    n_test: int
    marginal_d_coverage: float
    marginal_coverage_gap: float
    low_f_d_coverage: float
    mid_f_d_coverage: float
    high_f_d_coverage: float
    conditional_coverage_spread: float
    well_specified_conditional_spread: float
    aliasing_amplification: float
    low_f_d_bias: float
    high_f_d_bias: float
    marginal_preserved: bool
    conditional_aliasing_detected: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a JSON-compatible Python dictionary."""
        return {
            "family": str(self.family),
            "bvalues": list(self.bvalues),
            "snr": float(self.snr),
            "alpha": float(self.alpha),
            "nominal_coverage": float(self.nominal_coverage),
            "seed": int(self.seed),
            "n_cal": int(self.n_cal),
            "n_test": int(self.n_test),
            "marginal_d_coverage": float(self.marginal_d_coverage),
            "marginal_coverage_gap": float(self.marginal_coverage_gap),
            "low_f_d_coverage": float(self.low_f_d_coverage),
            "mid_f_d_coverage": float(self.mid_f_d_coverage),
            "high_f_d_coverage": float(self.high_f_d_coverage),
            "conditional_coverage_spread": float(self.conditional_coverage_spread),
            "well_specified_conditional_spread": float(self.well_specified_conditional_spread),
            "aliasing_amplification": float(self.aliasing_amplification),
            "low_f_d_bias": float(self.low_f_d_bias),
            "high_f_d_bias": float(self.high_f_d_bias),
            "marginal_preserved": bool(self.marginal_preserved),
            "conditional_aliasing_detected": bool(self.conditional_aliasing_detected),
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


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    rank = min(max(int(np.ceil((n + 1) * (1.0 - float(alpha)))), 1), n)
    return float(np.partition(s, rank - 1)[rank - 1])


def _simulate_heavy_tail_or_dispersion_clean(
    b: np.ndarray,
    d_arr: np.ndarray,
    ds_arr: np.ndarray,
    f_arr: np.ndarray,
    family: str,
    use_canonical_dro: bool,
) -> np.ndarray:
    """Generate clean non-bi-exponential decay with microvascular/tissue dispersion leakage into D."""
    fam = str(family).strip().lower()
    if fam == "dispersion_gamma":
        # Heavy microvascular velocity dispersion (shape k = 0.18) retains a slow algebraic tail
        # past b >= 200 even when mean D* is fast (22..55e-3), biasing D proportional to f while
        # clean bi-exponential control has negligible perfusion tail (< 1%) above b = 200.
        k_shape = 0.18
        perf = np.power(1.0 + b[None, :] * ds_arr[:, None] / k_shape, -k_shape)
        return f_arr[:, None] * perf + (1.0 - f_arr[:, None]) * np.exp(-b[None, :] * d_arr[:, None])
    if fam == "heavy_tail":
        k_shape = 0.15
        perf = np.power(1.0 + b[None, :] * ds_arr[:, None] / k_shape, -k_shape)
        return f_arr[:, None] * perf + (1.0 - f_arr[:, None]) * np.exp(-b[None, :] * d_arr[:, None])
    if fam == "stretched":
        perf = np.exp(-np.power(np.clip(b[None, :] * ds_arr[:, None], 0.0, None), 0.30))
        return f_arr[:, None] * perf + (1.0 - f_arr[:, None]) * np.exp(-b[None, :] * d_arr[:, None])
    phantom = generate_synthetic_phantom(
        b,
        D=d_arr,
        D_star=ds_arr,
        f=f_arr,
        snr=None,
        noise_type="none",
        family=fam,
        use_canonical_dro=use_canonical_dro,
    )
    return np.asarray(phantom.clean_signal, dtype=np.float64)


def _evaluate_conditional_d_coverage(
    b: np.ndarray,
    signals_cal: np.ndarray,
    signals_te: np.ndarray,
    d_cal: np.ndarray,
    d_te: np.ndarray,
    f_te: np.ndarray,
    alpha: float,
) -> tuple[float, float, float, float, float, float]:
    fit_cal = fit_ivim(b, signals_cal, method="segmented")
    fit_te = fit_ivim(b, signals_te, method="segmented")
    d_hat_cal = np.asarray(fit_cal.D, dtype=np.float64)
    d_hat_te = np.asarray(fit_te.D, dtype=np.float64)

    q_d = _conformal_quantile(np.abs(d_cal - d_hat_cal), alpha)
    lo = d_hat_te - q_d
    hi = d_hat_te + q_d
    covered = (d_te >= lo) & (d_te <= hi)
    marg_cov = float(np.mean(covered))

    edges = np.quantile(f_te, [1.0 / 3.0, 2.0 / 3.0])
    low_m = f_te <= edges[0]
    mid_m = (f_te > edges[0]) & (f_te <= edges[1])
    high_m = f_te > edges[1]

    cov_low = float(np.mean(covered[low_m]))
    cov_mid = float(np.mean(covered[mid_m]))
    cov_high = float(np.mean(covered[high_m]))
    bias_low = float(np.mean(d_hat_te[low_m] - d_te[low_m]))
    bias_high = float(np.mean(d_hat_te[high_m] - d_te[high_m]))
    return marg_cov, cov_low, cov_mid, cov_high, bias_low, bias_high


def probe_d_misspecification_aliasing(
    bvalues: np.ndarray | Sequence[float] | str | None = None,
    family: str = "dispersion_gamma",
    snr: float = 40.0,
    seed: int = 42,
    *,
    alpha: float = 0.10,
    n_cal: int = 240,
    n_test: int = 240,
    use_canonical_dro: bool = False,
) -> ProcrustesProbeReport:
    """Measure how bi-exponential misspecification preserves marginal D coverage while aliasing conditional D coverage."""
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
    seed_int = int(seed)
    rng = np.random.Generator(np.random.PCG64(seed_int))
    n_tot = int(n_cal) + int(n_test)

    d_all = rng.uniform(0.9e-3, 1.5e-3, size=n_tot)
    ds_all = rng.uniform(22.0e-3, 55.0e-3, size=n_tot)
    f_all = rng.uniform(0.02, 0.42, size=n_tot)

    clean_misspec = _simulate_heavy_tail_or_dispersion_clean(
        b, d_all, ds_all, f_all, family=family, use_canonical_dro=use_canonical_dro
    )
    clean_biexp = generate_synthetic_phantom(
        b, D=d_all, D_star=ds_all, f=f_all, snr=None, noise_type="none"
    ).clean_signal

    sigma = 1.0 / snr_val
    eps_re = rng.normal(0.0, sigma, size=clean_misspec.shape)
    eps_im = rng.normal(0.0, sigma, size=clean_misspec.shape)
    noisy_misspec = np.sqrt((clean_misspec + eps_re) ** 2 + eps_im**2)
    noisy_biexp = np.sqrt((clean_biexp + eps_re) ** 2 + eps_im**2)

    d_cal, d_te = d_all[:n_cal], d_all[n_cal:]
    f_te = f_all[n_cal:]

    marg_m, c_low_m, c_mid_m, c_high_m, b_low_m, b_high_m = _evaluate_conditional_d_coverage(
        b,
        noisy_misspec[:n_cal],
        noisy_misspec[n_cal:],
        d_cal,
        d_te,
        f_te,
        alpha=float(alpha),
    )
    _, c_low_w, c_mid_w, c_high_w, _, _ = _evaluate_conditional_d_coverage(
        b,
        noisy_biexp[:n_cal],
        noisy_biexp[n_cal:],
        d_cal,
        d_te,
        f_te,
        alpha=float(alpha),
    )

    spread_m = float(max(c_low_m, c_mid_m, c_high_m) - min(c_low_m, c_mid_m, c_high_m))
    spread_w = float(max(c_low_w, c_mid_w, c_high_w) - min(c_low_w, c_mid_w, c_high_w))
    nominal = 1.0 - float(alpha)
    gap = abs(marg_m - nominal)
    amp = spread_m / max(spread_w, 1e-3)

    return ProcrustesProbeReport(
        family=str(family),
        bvalues=tuple(float(x) for x in b),
        snr=snr_val,
        alpha=float(alpha),
        nominal_coverage=nominal,
        seed=seed_int,
        n_cal=int(n_cal),
        n_test=int(n_test),
        marginal_d_coverage=marg_m,
        marginal_coverage_gap=float(gap),
        low_f_d_coverage=c_low_m,
        mid_f_d_coverage=c_mid_m,
        high_f_d_coverage=c_high_m,
        conditional_coverage_spread=spread_m,
        well_specified_conditional_spread=spread_w,
        aliasing_amplification=float(amp),
        low_f_d_bias=b_low_m,
        high_f_d_bias=b_high_m,
        marginal_preserved=bool(gap <= 0.06),
        conditional_aliasing_detected=bool(spread_m >= 0.08 or abs(b_high_m - b_low_m) >= 5.0e-5),
    )
