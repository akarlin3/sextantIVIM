"""AveryCorp shared ``AuditReport`` diagnostic contract for ``sextant-ivim`` (Suite I, Stage 1).

Implements the deterministic empirical identifiability and boundary-railing gate
for **Suite I: Clinical Quantitative-MRI Trust Spine** (Stage 1: Forward NLLS
Fitting, Fisher CRLB & Assumption-Free :math:`D^*` Boundary-Railing Gate).

License: AGPL-3.0-or-later
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Final, Literal, Mapping

import numpy as np

from .fitting import compute_crlb, fit_ivim
from .t2_ivim import (
    DEFAULT_T2_IVIM_BOUNDS,
    DEFAULT_T2_IVIM_BVALS,
    DEFAULT_T2_IVIM_TES,
    RailingComparisonResult,
    compute_t2_ivim_crlb,
    fit_t2_ivim,
    t2_ivim_forward_model,
)

SuiteName = Literal[
    "Suite I: Clinical Quantitative-MRI Trust Spine",
    "Suite II: Scientific Identifiability & Null-Audit Suite",
]
AuditStatus = Literal["PASS", "WARN", "HALT"]
MetricValue = float | int | str | bool
ThresholdValue = float | int | str

PACKAGE_NAME: Final[str] = "sextant-ivim"
SUITE_I_NAME: Final[SuiteName] = "Suite I: Clinical Quantitative-MRI Trust Spine"
STAGE_1_ROLE: Final[str] = (
    "Stage 1: Forward NLLS Fitting, Fisher CRLB & Assumption-Free D* Boundary-Railing Gate"
)

AUDIT_SCHEMA_KEYS: Final[tuple[str, ...]] = (
    "package",
    "version",
    "suite",
    "stage_role",
    "status",
    "verdict",
    "metrics",
    "thresholds",
    "seed",
)

DEFAULT_AUDIT_THRESHOLDS: Final[dict[str, ThresholdValue]] = {
    "max_railing_fraction_2d": 0.25,
    "halt_railing_fraction_2d": 0.45,
    "max_median_crlb_cv_D": 0.20,
    "halt_median_crlb_cv_D": 0.40,
}


def _package_version() -> str:
    from . import __version__

    return str(__version__)


def _coerce_metric_scalar(key: str, value: Any) -> MetricValue:
    """Convert NumPy or Python scalar to a native Python ``float | int | str | bool``."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fval = float(value)
        if not np.isfinite(fval):
            raise ValueError(f"Metric {key!r} must be finite for JSON serialization; got {value!r}.")
        return fval
    if isinstance(value, str):
        return str(value)
    raise TypeError(
        f"Metric {key!r} must be float, int, str, or bool; got {type(value).__name__}."
    )


def _coerce_threshold_scalar(key: str, value: Any) -> ThresholdValue:
    """Convert NumPy or Python scalar to a native Python ``float | int | str`` (excluding ``bool``)."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Threshold {key!r} must be float, int, or str, not bool.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fval = float(value)
        if not np.isfinite(fval):
            raise ValueError(
                f"Threshold {key!r} must be finite for JSON serialization; got {value!r}."
            )
        return fval
    if isinstance(value, str):
        return str(value)
    raise TypeError(
        f"Threshold {key!r} must be float, int, or str; got {type(value).__name__}."
    )


def _evaluate_gate(
    railing_fraction_2d: float,
    median_crlb_cv_D: float,
    thresholds: Mapping[str, ThresholdValue],
) -> tuple[AuditStatus, str, str]:
    """Classify ``(status, verdict_code, verdict)`` from empirical railing and Fisher CRLB CV(D)."""
    max_rail = float(thresholds["max_railing_fraction_2d"])
    halt_rail = float(thresholds.get("halt_railing_fraction_2d", max(0.45, max_rail)))
    max_cv_d = float(thresholds["max_median_crlb_cv_D"])
    halt_cv_d = float(thresholds.get("halt_median_crlb_cv_D", max(0.40, max_cv_d)))

    if railing_fraction_2d <= max_rail and median_crlb_cv_D <= max_cv_d:
        verdict_code = "INTERIOR_IDENTIFIABLE"
        verdict = (
            f"INTERIOR_IDENTIFIABLE: 2D (b, TE) relaxation-compensated fitting suppresses "
            f"D* boundary railing to {railing_fraction_2d:.1%} (<= {max_rail:.0%}) with "
            f"well-conditioned parenchymal D CRLB CV {median_crlb_cv_D:.3f} (<= {max_cv_d:.2f})."
        )
        return "PASS", verdict_code, verdict

    verdict_code = "BOUNDARY_RAILED_WALL"
    if railing_fraction_2d >= halt_rail or median_crlb_cv_D >= halt_cv_d:
        status: AuditStatus = "HALT"
        action = "halt uncalibrated point-estimate inference and escalate to Stage 2 conformal intervals"
    else:
        status = "WARN"
        action = "flag marginal pseudo-diffusion identifiability for Stage 2 conformal calibration"

    verdict = (
        f"BOUNDARY_RAILED_WALL: D* boundary railing ({railing_fraction_2d:.1%} vs max "
        f"{max_rail:.0%}) or parenchymal D CRLB CV ({median_crlb_cv_D:.3f} vs max "
        f"{max_cv_d:.2f}) breached gate thresholds; {action}."
    )
    return status, verdict_code, verdict


@dataclass(frozen=True)
class AuditReport:
    """Immutable AveryCorp standardized diagnostic report for ``sextant-ivim`` (Suite I, Stage 1)."""

    package: str = PACKAGE_NAME
    version: str = field(default_factory=_package_version)
    suite: SuiteName = SUITE_I_NAME
    stage_role: str = STAGE_1_ROLE
    status: AuditStatus = "PASS"
    verdict: str = (
        "INTERIOR_IDENTIFIABLE: 2D (b, TE) relaxation-compensated fitting suppresses "
        "D* boundary railing below threshold with well-conditioned D CRLB."
    )
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    thresholds: dict[str, ThresholdValue] = field(
        default_factory=lambda: dict(DEFAULT_AUDIT_THRESHOLDS)
    )
    seed: int = 42

    def __post_init__(self) -> None:
        clean_metrics = {
            str(k): _coerce_metric_scalar(str(k), v) for k, v in self.metrics.items()
        }
        clean_thresholds = {
            str(k): _coerce_threshold_scalar(str(k), v) for k, v in self.thresholds.items()
        }
        object.__setattr__(self, "package", str(self.package))
        object.__setattr__(self, "version", str(self.version))
        object.__setattr__(self, "stage_role", str(self.stage_role))
        object.__setattr__(self, "verdict", str(self.verdict))
        object.__setattr__(self, "metrics", clean_metrics)
        object.__setattr__(self, "thresholds", clean_thresholds)
        object.__setattr__(self, "seed", int(self.seed))

    def to_dict(self) -> dict[str, Any]:
        """Serialize this ``AuditReport`` to a plain JSON-safe dictionary with all 9 schema keys."""
        return {
            "package": str(self.package),
            "version": str(self.version),
            "suite": str(self.suite),
            "stage_role": str(self.stage_role),
            "status": str(self.status),
            "verdict": str(self.verdict),
            "metrics": {
                str(k): _coerce_metric_scalar(str(k), v) for k, v in self.metrics.items()
            },
            "thresholds": {
                str(k): _coerce_threshold_scalar(str(k), v) for k, v in self.thresholds.items()
            },
            "seed": int(self.seed),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize this ``AuditReport`` to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary_line(self) -> str:
        """Return a concise single-line human-readable diagnostic summary."""
        r1 = float(self.metrics.get("railing_fraction_1d", 0.0))
        r2 = float(self.metrics.get("railing_fraction_2d", 0.0))
        cv_d = float(self.metrics.get("median_crlb_cv_D", 0.0))
        return (
            f"[{self.status}] {self.package} v{self.version} | {self.suite} | "
            f"seed={self.seed} | 1D_rail={r1:.1%} -> 2D_rail={r2:.1%} | "
            f"median_CRLB_CV(D)={cv_d:.3f} | {self.verdict}"
        )


def build_audit_report_from_comparison(
    cmp_res: RailingComparisonResult,
    *,
    thresholds: Mapping[str, ThresholdValue] | None = None,
) -> AuditReport:
    """Construct a standardized :class:`AuditReport` from a :class:`RailingComparisonResult`."""
    thresh_dict: dict[str, ThresholdValue] = (
        dict(DEFAULT_AUDIT_THRESHOLDS)
        if thresholds is None
        else {str(k): _coerce_threshold_scalar(str(k), v) for k, v in thresholds.items()}
    )
    rail_1d = float(cmp_res.railing_rate_1d)
    rail_2d = float(cmp_res.railing_rate_2d)
    cv_d_2d = float(cmp_res.crlb_2d["cv"]["D"])
    cv_ds_2d = float(cmp_res.crlb_2d["cv"]["D_star"])
    seed_val = int(cmp_res.seed) if cmp_res.seed is not None else 42

    status, verdict_code, verdict = _evaluate_gate(rail_2d, cv_d_2d, thresh_dict)
    metrics: dict[str, MetricValue] = {
        "n_voxels": int(cmp_res.n_trials),
        "snr": float(cmp_res.snr),
        "railing_fraction_1d": rail_1d,
        "railing_fraction_2d": rail_2d,
        "railing_reduction": float(cmp_res.railing_reduction),
        "railing_relative_reduction": float(cmp_res.railing_relative_reduction),
        "median_crlb_cv_D": cv_d_2d,
        "median_crlb_cv_D_star": cv_ds_2d,
        "f_bias_1d": float(cmp_res.f_bias_1d),
        "f_bias_2d": float(cmp_res.f_bias_2d),
        "d_star_rmse_1d": float(cmp_res.d_star_rmse_1d),
        "d_star_rmse_2d": float(cmp_res.d_star_rmse_2d),
        "fim_condition_number_2d": float(cmp_res.crlb_2d["condition_number"]),
        "interior_identifiable": bool(status == "PASS"),
        "verdict_code": verdict_code,
    }
    return AuditReport(
        package=PACKAGE_NAME,
        version=_package_version(),
        suite=SUITE_I_NAME,
        stage_role=STAGE_1_ROLE,
        status=status,
        verdict=verdict,
        metrics=metrics,
        thresholds=thresh_dict,
        seed=seed_val,
    )


def run_audit(
    seed: int = 42,
    *,
    n_voxels: int = 48,
    snr: float = 65.0,
) -> AuditReport:
    """Run the deterministic Suite I Stage 1 ``sextant-ivim`` empirical identifiability audit.

    Generates a deterministic (``PCG64``) heterogeneous synthetic parenchymal cohort
    of ``n_voxels`` voxels under scan-time-matched 1D single-TE ``S(b)`` and 2D
    multi-echo ``S(b, TE)`` schedules, fits both models via bounded NLLS, evaluates
    the assumption-free :math:`D^*` boundary-railing rates alongside exact per-voxel
    5x5 Fisher CRLB coefficients of variation :math:`\\sigma_{\\text{CRLB}}(D) / D`,
    and classifies regime identifiability against Stage 1 trust thresholds.

    Parameters
    ----------
    seed : int, default=42
        Deterministic ``PCG64`` pseudo-random generator seed.
    n_voxels : int, default=48
        Number of synthetic parenchymal voxels evaluated in the audit cohort.
    snr : float, default=65.0
        Baseline proton-density thermal SNR at :math:`(b = 0, \\text{TE} = 0)`.

    Returns
    -------
    AuditReport
        Frozen standardized diagnostic report conforming to the AveryCorp contract.
    """
    if int(n_voxels) < 1:
        raise ValueError(f"`n_voxels` must be >= 1; got {n_voxels!r}.")
    snr_val = float(snr)
    if snr_val <= 0.0 or not np.isfinite(snr_val):
        raise ValueError(f"`snr` must be positive and finite; got {snr!r}.")

    seed_int = int(seed)
    n_vox = int(n_voxels)
    rng = np.random.Generator(np.random.PCG64(seed_int))

    b_arr = DEFAULT_T2_IVIM_BVALS.copy()
    te_arr = DEFAULT_T2_IVIM_TES.copy()
    te_1d = 85.0
    nex_1d = int(te_arr.size)
    s0_true = 1.0
    sigma_shot = s0_true / snr_val

    # Heterogeneous physiological parenchymal tissue cohort (venous/capillary T2_p < parenchymal T2_t)
    d_true = rng.uniform(0.90e-3, 1.30e-3, size=n_vox)
    ds_true = rng.uniform(0.020, 0.030, size=n_vox)
    f_true = rng.uniform(0.14, 0.20, size=n_vox)
    t2p_true = rng.uniform(40.0, 50.0, size=n_vox)
    t2t_true = rng.uniform(88.0, 102.0, size=n_vox)

    clean_2d = t2_ivim_forward_model(
        b_arr,
        te_arr,
        D=d_true,
        D_star=ds_true,
        f=f_true,
        T2_p=t2p_true,
        T2_t=t2t_true,
        S0=s0_true,
        grid=True,
    )
    clean_1d = t2_ivim_forward_model(
        b_arr,
        np.full_like(b_arr, te_1d),
        D=d_true,
        D_star=ds_true,
        f=f_true,
        T2_p=t2p_true,
        T2_t=t2t_true,
        S0=s0_true,
        grid=False,
    )

    eps_2d = rng.normal(0.0, sigma_shot, size=(n_vox, b_arr.size, te_arr.size))
    noisy_2d = np.clip(clean_2d + eps_2d, 1e-12, None)

    eps_1d = rng.normal(0.0, sigma_shot, size=(n_vox, nex_1d, b_arr.size))
    noisy_1d = np.clip(np.mean(clean_1d[:, None, :] + eps_1d, axis=1), 1e-12, None)

    bounds_dict = dict(DEFAULT_T2_IVIM_BOUNDS)
    fit_1d = fit_ivim(
        b_arr,
        noisy_1d,
        method="nlls",
        bounds_D=bounds_dict["D"],
        bounds_D_star=bounds_dict["D_star"],
        bounds_f=bounds_dict["f"],
    )
    fit_2d = fit_t2_ivim(
        noisy_2d,
        b_arr,
        te_arr,
        bounds=bounds_dict,
        grid=True,
    )

    rail_1d = float(fit_1d.frac_railed)
    rail_2d = float(fit_2d.frac_railed)
    rail_reduction = rail_1d - rail_2d
    rail_rel_reduction = rail_reduction / max(rail_1d, 1e-12) if rail_1d > 0.0 else 0.0

    # Per-voxel analytical Fisher CRLB across the synthetic parenchymal cohort
    crlb_cv_d_2d = np.empty(n_vox, dtype=np.float64)
    crlb_cv_ds_2d = np.empty(n_vox, dtype=np.float64)
    crlb_cond_2d = np.empty(n_vox, dtype=np.float64)
    crlb_cv_d_1d = np.empty(n_vox, dtype=np.float64)

    sigma_eff_1d = sigma_shot / float(np.sqrt(nex_1d))
    for idx in range(n_vox):
        c2 = compute_t2_ivim_crlb(
            b_arr,
            te_arr,
            D=float(d_true[idx]),
            D_star=float(ds_true[idx]),
            f=float(f_true[idx]),
            T2_p=float(t2p_true[idx]),
            T2_t=float(t2t_true[idx]),
            S0=s0_true,
            sigma=sigma_shot,
            grid=True,
        )
        crlb_cv_d_2d[idx] = float(c2["cv"]["D"])
        crlb_cv_ds_2d[idx] = float(c2["cv"]["D_star"])
        crlb_cond_2d[idx] = float(c2["condition_number"])

        w_p = float(f_true[idx]) * float(np.exp(-te_1d / float(t2p_true[idx])))
        w_t = (1.0 - float(f_true[idx])) * float(np.exp(-te_1d / float(t2t_true[idx])))
        s0_eff = s0_true * (w_p + w_t)
        f_app = w_p / max(w_p + w_t, 1e-30)
        c1 = compute_crlb(
            b_arr,
            D=float(d_true[idx]),
            D_star=float(ds_true[idx]),
            f=f_app,
            snr=s0_eff / sigma_eff_1d,
            S0=s0_eff,
        )
        crlb_cv_d_1d[idx] = float(np.asarray(c1["std"])[0] / float(d_true[idx]))

    median_crlb_cv_d = float(np.median(crlb_cv_d_2d))
    median_crlb_cv_ds = float(np.median(crlb_cv_ds_2d))
    median_crlb_cv_d_1d = float(np.median(crlb_cv_d_1d))
    median_cond_2d = float(np.median(crlb_cond_2d))

    f_bias_1d = float(np.nanmean(np.asarray(fit_1d.f, dtype=np.float64) - f_true))
    f_bias_2d = float(np.nanmean(np.asarray(fit_2d.f, dtype=np.float64) - f_true))
    ds_rmse_1d = float(
        np.sqrt(np.nanmean((np.asarray(fit_1d.D_star, dtype=np.float64) - ds_true) ** 2))
    )
    ds_rmse_2d = float(
        np.sqrt(np.nanmean((np.asarray(fit_2d.D_star, dtype=np.float64) - ds_true) ** 2))
    )

    thresh_dict: dict[str, ThresholdValue] = dict(DEFAULT_AUDIT_THRESHOLDS)
    status, verdict_code, verdict = _evaluate_gate(rail_2d, median_crlb_cv_d, thresh_dict)

    metrics: dict[str, MetricValue] = {
        "n_voxels": n_vox,
        "snr": snr_val,
        "railing_fraction_1d": rail_1d,
        "railing_fraction_2d": rail_2d,
        "railing_reduction": rail_reduction,
        "railing_relative_reduction": rail_rel_reduction,
        "median_crlb_cv_D": median_crlb_cv_d,
        "median_crlb_cv_D_1d": median_crlb_cv_d_1d,
        "median_crlb_cv_D_star": median_crlb_cv_ds,
        "f_bias_1d": f_bias_1d,
        "f_bias_2d": f_bias_2d,
        "d_star_rmse_1d": ds_rmse_1d,
        "d_star_rmse_2d": ds_rmse_2d,
        "fim_condition_number_2d": median_cond_2d,
        "interior_identifiable": bool(status == "PASS"),
        "verdict_code": verdict_code,
    }

    return AuditReport(
        package=PACKAGE_NAME,
        version=_package_version(),
        suite=SUITE_I_NAME,
        stage_role=STAGE_1_ROLE,
        status=status,
        verdict=verdict,
        metrics=metrics,
        thresholds=thresh_dict,
        seed=seed_int,
    )
