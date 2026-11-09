"""Deterministic unit tests for conservation laws, numerical convergence, CRLB, and calibration."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import sextant_core
from sextant_core import (
    GLOBAL_SEED,
    BootCI,
    CalibrationSummary,
    bootstrap_fraction,
    bootstrap_railing_ci,
    classify_railing,
    compute_crlb,
    compute_jacobian,
    empirical_coverage,
    evaluate_calibration,
    expected_calibration_error,
    fit_ivim,
    fit_nlls_voxel,
    fit_segmented_voxel,
    generate_synthetic_phantom,
    ivim_forward_model,
    kurtosis_forward_model,
    make_rng,
    relative_sharpness,
    requires_ground_truth,
    summarize_df_context,
    triexp_forward_model,
)


def test_package_version_and_py_typed_marker():
    """Verify package version 1.0.0 and PEP 561 py.typed marker presence."""
    assert sextant_core.__version__ == "1.0.0"
    pkg_dir = Path(os.path.dirname(sextant_core.__file__))
    assert (pkg_dir / "py.typed").exists()


def test_conservation_laws_and_compartment_invariants():
    """Verify physical conservation laws and limiting reduction invariants of S(b)."""
    bvals = np.array([0.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0], dtype=np.float64)
    s0 = 250.0
    d_val = 1.1e-3
    ds_val = 2.5e-2

    # 1. Baseline normalization invariant: S(b=0) == S0 * (f + (1 - f)) == S0 for all f in [0, 1]
    for f_val in (0.0, 0.05, 0.15, 0.50, 1.0):
        sig = ivim_forward_model(bvals, D=d_val, D_star=ds_val, f=f_val, S0=s0)
        assert np.isclose(sig[0], s0, rtol=0.0, atol=1e-12)
        # Strict monotonicity: dS/db < 0 for b > 0
        assert np.all(np.diff(sig) < 0.0)
        assert np.all((sig > 0.0) & (sig <= s0))

    # 2. Pure parenchymal limit (f = 0): independent of D_star
    sig_f0_a = ivim_forward_model(bvals, D=d_val, D_star=0.01, f=0.0, S0=s0)
    sig_f0_b = ivim_forward_model(bvals, D=d_val, D_star=0.12, f=0.0, S0=s0)
    mono_tissue = s0 * np.exp(-bvals * d_val)
    assert np.allclose(sig_f0_a, mono_tissue, rtol=0.0, atol=1e-13)
    assert np.allclose(sig_f0_b, mono_tissue, rtol=0.0, atol=1e-13)

    # 3. Pure microvascular limit (f = 1): independent of D
    sig_f1_a = ivim_forward_model(bvals, D=0.5e-3, D_star=ds_val, f=1.0, S0=s0)
    sig_f1_b = ivim_forward_model(bvals, D=3.0e-3, D_star=ds_val, f=1.0, S0=s0)
    mono_perf = s0 * np.exp(-bvals * ds_val)
    assert np.allclose(sig_f1_a, mono_perf, rtol=0.0, atol=1e-13)
    assert np.allclose(sig_f1_b, mono_perf, rtol=0.0, atol=1e-13)

    # 4. Equal-diffusivity degeneracy (D_star == D): independent of f
    for f_val in (0.0, 0.2, 0.7, 1.0):
        sig_deg = ivim_forward_model(bvals, D=d_val, D_star=d_val, f=f_val, S0=s0)
        assert np.allclose(sig_deg, mono_tissue, rtol=0.0, atol=1e-13)

    # 5. Continuum reduction of tri-exponential (w=0) and kurtosis (K=0) models
    sig_bi = ivim_forward_model(bvals, D=d_val, D_star=ds_val, f=0.18, S0=s0)
    sig_tri0 = triexp_forward_model(bvals, D=d_val, D_star=ds_val, f=0.18, S0=s0, w=0.0)
    sig_kurt0 = kurtosis_forward_model(bvals, D=d_val, D_star=ds_val, f=0.18, S0=s0, K=0.0)
    assert np.allclose(sig_tri0, sig_bi, rtol=0.0, atol=1e-13)
    assert np.allclose(sig_kurt0, sig_bi, rtol=0.0, atol=1e-13)


def test_scale_invariance_of_fitted_parameters():
    """Verify multiplying signal by alpha > 0 leaves (D, D_star, f) invariant and scales S0."""
    phantom = generate_synthetic_phantom(D=0.0012, D_star=0.022, f=0.16, S0=1.0, snr=None)
    res_unit = fit_ivim(phantom.bvals, phantom.signal, method="nlls")
    alpha = 1500.0
    res_scaled = fit_ivim(phantom.bvals, alpha * phantom.signal, method="nlls")

    assert np.isclose(res_scaled.D, res_unit.D, rtol=1e-6, atol=1e-8)
    assert np.isclose(res_scaled.D_star, res_unit.D_star, rtol=1e-5, atol=1e-7)
    assert np.isclose(res_scaled.f, res_unit.f, rtol=1e-5, atol=1e-7)
    assert np.isclose(res_scaled.S0, alpha * res_unit.S0, rtol=1e-9, atol=1e-9)


def test_monotonic_snr_convergence_with_fixed_seed():
    """Verify parameter recovery RMSE decreases monotonically as SNR increases."""
    d_true, ds_true, f_true = 0.0011, 0.020, 0.18
    rmses_d: list[float] = []
    for snr in (20.0, 60.0, 200.0):
        ph = generate_synthetic_phantom(
            D=d_true,
            D_star=ds_true,
            f=f_true,
            S0=100.0,
            snr=snr,
            noise_type="gaussian",
            n_voxels=64,
            seed=GLOBAL_SEED,
        )
        fit = fit_ivim(ph.bvals, ph.signal, method="segmented")
        rmse_d = float(np.sqrt(np.mean((np.asarray(fit.D) - d_true) ** 2)))
        rmses_d.append(rmse_d)

    assert rmses_d[0] > rmses_d[1] > rmses_d[2]


def test_analytical_jacobian_and_crlb_scaling():
    """Verify analytical Jacobian matches finite differences and CRLB scales as 1/SNR."""
    bvals = np.array([0.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0], dtype=np.float64)
    d_val, ds_val, f_val, s0_val = 0.0012, 0.025, 0.15, 2.0
    jac = compute_jacobian(bvals, D=d_val, D_star=ds_val, f=f_val, S0=s0_val)
    assert jac.shape == (bvals.size, 3)

    eps = 1e-8
    fd_d = (
        ivim_forward_model(bvals, d_val + eps, ds_val, f_val, s0_val)
        - ivim_forward_model(bvals, d_val - eps, ds_val, f_val, s0_val)
    ) / (2.0 * eps)
    fd_ds = (
        ivim_forward_model(bvals, d_val, ds_val + eps, f_val, s0_val)
        - ivim_forward_model(bvals, d_val, ds_val - eps, f_val, s0_val)
    ) / (2.0 * eps)
    fd_f = (
        ivim_forward_model(bvals, d_val, ds_val, f_val + eps, s0_val)
        - ivim_forward_model(bvals, d_val, ds_val, f_val - eps, s0_val)
    ) / (2.0 * eps)

    assert np.allclose(jac[:, 0], fd_d, rtol=1e-6, atol=1e-8)
    assert np.allclose(jac[:, 1], fd_ds, rtol=1e-6, atol=1e-8)
    assert np.allclose(jac[:, 2], fd_f, rtol=1e-6, atol=1e-8)

    crlb_25 = compute_crlb(bvals, D=d_val, D_star=ds_val, f=f_val, snr=25.0, reg=0.0)
    crlb_50 = compute_crlb(bvals, D=d_val, D_star=ds_val, f=f_val, snr=50.0, reg=0.0)
    assert np.allclose(np.asarray(crlb_25["std"]) / np.asarray(crlb_50["std"]), 2.0, rtol=1e-5)
    assert float(crlb_25["cv_D_star"]) > float(crlb_50["cv_D_star"]) > 0.0


def test_edge_case_stability_and_validation():
    """Verify edge-case handling for NaN/non-positive signals and invalid arguments."""
    bvals = np.array([0.0, 50.0, 200.0, 600.0], dtype=np.float64)

    # All-zero or NaN signals return (nan, nan, nan) without crashing
    for bad_sig in (np.zeros(4), np.full(4, -1.0), np.array([1.0, np.nan, 0.5, 0.2])):
        d_seg, ds_seg, f_seg = fit_segmented_voxel(bvals, bad_sig)
        assert np.isnan(d_seg) and np.isnan(ds_seg) and np.isnan(f_seg)
        d_nlls, ds_nlls, f_nlls = fit_nlls_voxel(bvals, bad_sig)
        assert np.isnan(d_nlls) and np.isnan(ds_nlls) and np.isnan(f_nlls)

    # Non-zero minimum b-value schedule (e.g. clinical 3-b schedule [50, 400, 800])
    b_sparse = np.array([50.0, 400.0, 800.0], dtype=np.float64)
    sig_sparse = ivim_forward_model(b_sparse, D=0.0011, D_star=0.02, f=0.15, S0=500.0)
    res_sparse = fit_ivim(b_sparse, sig_sparse, method="segmented")
    assert np.isfinite(res_sparse.D) and np.isfinite(res_sparse.D_star) and np.isfinite(res_sparse.f)

    # Validation errors on mismatched shapes or unsupported methods
    with pytest.raises(ValueError):
        fit_ivim(bvals, np.ones(3))
    with pytest.raises(ValueError):
        fit_ivim(bvals, np.ones(4), method="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        generate_synthetic_phantom(snr=-10.0)
    with pytest.raises(ValueError):
        compute_crlb(bvals, D=0.001, D_star=0.02, f=0.15, snr=0.0)


def test_calibration_utilities_and_bootstrap_determinism():
    """Verify calibration ruler and bootstrap CI utilities with fixed seed."""
    assert requires_ground_truth() is True
    rng = make_rng(GLOBAL_SEED)
    truth = 0.025
    sigma = 0.004
    estimates = truth + rng.normal(0.0, sigma, size=10000)

    summary = evaluate_calibration(estimates, truth, sigma)
    assert isinstance(summary, CalibrationSummary)
    assert summary.n_samples == 10000
    assert summary.ece < 0.015
    assert np.isclose(summary.ece, expected_calibration_error(empirical_coverage(estimates, truth, sigma)))
    assert np.isclose(summary.sharpness_rel, relative_sharpness(sigma, truth, level=0.95))

    railed = rng.random(1000) < 0.547
    ci1 = bootstrap_fraction(railed, n_boot=1000, seed=GLOBAL_SEED)
    ci2 = bootstrap_fraction(railed, n_boot=1000, seed=GLOBAL_SEED)
    assert isinstance(ci1, BootCI)
    assert ci1 == ci2
    assert ci1.lo <= ci1.point <= ci1.hi

    pct, lo_pct, hi_pct = bootstrap_railing_ci(railed, n_boot=1000, seed=GLOBAL_SEED)
    assert 0.0 <= lo_pct <= pct <= hi_pct <= 100.0

    d_arr = rng.uniform(0.8e-3, 1.5e-3, size=1000)
    f_arr = rng.uniform(0.05, 0.30, size=1000)
    rail_dict = classify_railing(np.where(railed, 0.15, 0.02))
    ctx = summarize_df_context(d_arr, f_arr, rail_dict["railed_mask"])  # type: ignore[arg-type]
    assert set(ctx.keys()) == {"all", "railed", "interior"}
