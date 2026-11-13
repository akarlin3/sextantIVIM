"""Sextant Core — OSIPI-ready 1D and 2D (b, TE) IVIM-DWI fitting, boundary-railing diagnostics, NIfTI volume processing, and calibration utilities.

Exposes the primary public API:
- :func:`fit_ivim` — two-step segmented and simultaneous bounded NLLS IVIM fitting
  for 1D, 2D, 3D, and NIfTI diffusion-weighted imaging data, with automatic
  boundary-railing classification for :math:`D^*`.
- :func:`t2_ivim_forward_model`, :func:`fit_t2_ivim`, :func:`compute_t2_ivim_crlb`,
  :func:`compare_1d_vs_2d_railing` — two-compartment 2D ``(b, TE)`` relaxation-compensated
  IVIM modeling, bounded NLLS fitting, exact 5x5 CRLB identifiability analysis,
  and 1D vs 2D boundary-railing diagnostics.
- :func:`read_nifti`, :func:`write_nifti`, :func:`read_bvals`, :func:`write_bvals`,
  :func:`fit_volume`, :func:`fit_volume_files` — pure-NumPy/nibabel NIfTI-1 I/O
  and multi-core chunked volumetric IVIM fitting.
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

__version__ = "1.2.0"

from .audit import (
    AuditAttestation,
    AuditReport,
    build_audit_attestation,
    compute_input_sha256,
    compute_platform_fingerprint,
    compute_report_sha256,
    run_audit,
)
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
    has_ivimbench_phantoms,
    ivim_forward_model,
    kurtosis_forward_model,
    resolve_clinical_bvals,
    triexp_forward_model,
)
from .procrustes_probe import (
    ProcrustesProbeReport,
    probe_d_misspecification_aliasing,
)
from .t2_ivim import (
    DEFAULT_BOUNDS_T2_P,
    DEFAULT_BOUNDS_T2_T,
    DEFAULT_T2_IVIM_BOUNDS,
    DEFAULT_T2_IVIM_BVALS,
    DEFAULT_T2_IVIM_TES,
    RailingComparisonResult,
    T2IVIMFitResult,
    compare_1d_vs_2d_railing,
    compute_t2_ivim_crlb,
    compute_t2_ivim_jacobian,
    fit_t2_ivim,
    fit_t2_ivim_voxel,
    t2_ivim_forward_model,
)
from .vernier_design import (
    VernierSchemeReport,
    build_vernier_nex_weights,
    evaluate_calibration_aware_scheme,
)
from .volume import (
    NiftiVolume,
    VolumeFitResult,
    fit_volume,
    fit_volume_files,
    read_bvals,
    read_nifti,
    read_te_values,
    write_bvals,
    write_nifti,
    write_te_values,
)

__all__ = [
    "AuditAttestation",
    "AuditReport",
    "BootCI",
    "CalibrationSummary",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOUNDS_D",
    "DEFAULT_BOUNDS_D_STAR",
    "DEFAULT_BOUNDS_F",
    "DEFAULT_BOUNDS_T2_P",
    "DEFAULT_BOUNDS_T2_T",
    "DEFAULT_BVALS",
    "DEFAULT_B_THRESH",
    "DEFAULT_CALIBRATION_LEVELS",
    "DEFAULT_PHANTOM_BVALS",
    "DEFAULT_RAIL_TOL",
    "DEFAULT_T2_IVIM_BOUNDS",
    "DEFAULT_T2_IVIM_BVALS",
    "DEFAULT_T2_IVIM_TES",
    "GLOBAL_SEED",
    "IVIMFitResult",
    "NiftiVolume",
    "ProcrustesProbeReport",
    "RailingComparisonResult",
    "SyntheticPhantom",
    "T2IVIMFitResult",
    "VernierSchemeReport",
    "VolumeFitResult",
    "bootstrap_fraction",
    "bootstrap_railing_ci",
    "build_audit_attestation",
    "build_vernier_nex_weights",
    "classify_railing",
    "compare_1d_vs_2d_railing",
    "compute_crlb",
    "compute_input_sha256",
    "compute_jacobian",
    "compute_platform_fingerprint",
    "compute_report_sha256",
    "compute_t2_ivim_crlb",
    "compute_t2_ivim_jacobian",
    "coverage",
    "ece",
    "empirical_coverage",
    "evaluate_calibration",
    "evaluate_calibration_aware_scheme",
    "expected_calibration_error",
    "fit_ivim",
    "fit_nlls_voxel",
    "fit_segmented_voxel",
    "fit_t2_ivim",
    "fit_t2_ivim_voxel",
    "fit_volume",
    "fit_volume_files",
    "generate_synthetic_phantom",
    "has_ivimbench_phantoms",
    "ivim_forward_model",
    "kurtosis_forward_model",
    "make_rng",
    "probe_d_misspecification_aliasing",
    "read_bvals",
    "read_nifti",
    "read_te_values",
    "relative_sharpness",
    "requires_ground_truth",
    "resolve_clinical_bvals",
    "run_audit",
    "sharpness_rel",
    "summarize_df_context",
    "t2_ivim_forward_model",
    "triexp_forward_model",
    "write_bvals",
    "write_nifti",
    "write_te_values",
]
