"""Sextant Core — OSIPI-ready bi-exponential IVIM-DWI fitting, boundary-railing diagnostics, and calibration utilities.

Exposes the primary public API:
- :func:`fit_ivim` — two-step segmented and simultaneous bounded NLLS IVIM fitting
  for 1D, 2D, 3D, and NIfTI diffusion-weighted imaging data, with automatic
  boundary-railing classification for :math:`D^*`.
- :func:`generate_synthetic_phantom` — 1D/2D synthetic bi-exponential IVIM signal
  generator with configurable Rician or Gaussian noise.
- :func:`ivim_forward_model`, :func:`triexp_forward_model`, :func:`kurtosis_forward_model` —
  bi-exponential, tri-exponential, and diffusional-kurtosis decay models.
- :func:`compute_jacobian`, :func:`compute_crlb` — analytical Jacobian and
  Cramér–Rao Lower Bound (CRLB) identifiability diagnostics.
- :func:`evaluate_calibration`, :func:`empirical_coverage`, :func:`expected_calibration_error`,
  :func:`relative_sharpness`, :func:`bootstrap_fraction`, :func:`bootstrap_railing_ci`,
  :func:`summarize_df_context` — calibration ruler and bootstrap inference utilities.
"""
from __future__ import annotations

from .calibration import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CALIBRATION_LEVELS,
    GLOBAL_SEED,
    BootCI,
    CalibrationSummary,
    bootstrap_fraction,
    bootstrap_railing_ci,
    coverage,
    ece,
    empirical_coverage,
    evaluate_calibration,
    expected_calibration_error,
    make_rng,
    relative_sharpness,
    requires_ground_truth,
    sharpness_rel,
    summarize_df_context,
)
from .fitting import (
    DEFAULT_B_THRESH,
    DEFAULT_BOUNDS_D,
    DEFAULT_BOUNDS_D_STAR,
    DEFAULT_BOUNDS_F,
    DEFAULT_BVALS,
    DEFAULT_RAIL_TOL,
    IVIMFitResult,
    classify_railing,
    compute_crlb,
    compute_jacobian,
    fit_ivim,
    fit_nlls_voxel,
    fit_segmented_voxel,
)
from .phantom import (
    DEFAULT_PHANTOM_BVALS,
    SyntheticPhantom,
    generate_synthetic_phantom,
    ivim_forward_model,
    kurtosis_forward_model,
    triexp_forward_model,
)

__all__ = [
    "BootCI",
    "CalibrationSummary",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOUNDS_D",
    "DEFAULT_BOUNDS_D_STAR",
    "DEFAULT_BOUNDS_F",
    "DEFAULT_BVALS",
    "DEFAULT_B_THRESH",
    "DEFAULT_CALIBRATION_LEVELS",
    "DEFAULT_PHANTOM_BVALS",
    "DEFAULT_RAIL_TOL",
    "GLOBAL_SEED",
    "IVIMFitResult",
    "SyntheticPhantom",
    "bootstrap_fraction",
    "bootstrap_railing_ci",
    "classify_railing",
    "compute_crlb",
    "compute_jacobian",
    "coverage",
    "ece",
    "empirical_coverage",
    "evaluate_calibration",
    "expected_calibration_error",
    "fit_ivim",
    "fit_nlls_voxel",
    "fit_segmented_voxel",
    "generate_synthetic_phantom",
    "ivim_forward_model",
    "kurtosis_forward_model",
    "make_rng",
    "relative_sharpness",
    "requires_ground_truth",
    "sharpness_rel",
    "summarize_df_context",
    "triexp_forward_model",
]
__version__ = "1.0.0"
