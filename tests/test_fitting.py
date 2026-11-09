"""Automated pytest suite for sextant_core IVIM fitting and synthetic phantom recovery."""
from __future__ import annotations

import numpy as np
import pytest

from sextant_core import (
    IVIMFitResult,
    SyntheticPhantom,
    classify_railing,
    fit_ivim,
    generate_synthetic_phantom,
)


def test_fit_ivim_recovers_clean_1d_ground_truth():
    """Verify fit_ivim() recovers (D=0.001, D_star=0.02, f=0.15) on a noise-free 1D curve."""
    d_true = 0.001
    d_star_true = 0.02
    f_true = 0.15

    phantom = generate_synthetic_phantom(
        D=d_true,
        D_star=d_star_true,
        f=f_true,
        S0=1.0,
        snr=None,
        noise_type="none",
    )

    for method in ("segmented", "nlls"):
        res = fit_ivim(phantom.bvals, phantom.signal, method=method, b_thresh=200.0)
        assert isinstance(res, IVIMFitResult)
        assert np.isclose(res.D, d_true, rtol=0.02, atol=5e-5)
        assert np.isclose(res.D_star, d_star_true, rtol=0.05, atol=1e-3)
        assert np.isclose(res.f, f_true, rtol=0.05, atol=1e-2)
        assert res.railed_mask is False


@pytest.mark.parametrize("noise_type", ["rician", "gaussian"])
def test_fit_ivim_recovers_noisy_1d_and_2d_phantom(noise_type: str):
    """Verify fit_ivim() recovers (D=0.001, D_star=0.02, f=0.15) under Rician & Gaussian noise."""
    d_true = 0.001
    d_star_true = 0.02
    f_true = 0.15

    phantom = generate_synthetic_phantom(
        D=d_true,
        D_star=d_star_true,
        f=f_true,
        S0=100.0,
        snr=60.0,
        noise_type=noise_type,
        shape=(8, 8),
        seed=20260613,
    )
    assert isinstance(phantom, SyntheticPhantom)
    assert phantom.signal.shape == (8, 8, phantom.bvals.size)

    res = fit_ivim(phantom.bvals, phantom.signal, method="segmented", b_thresh=200.0)
    assert res.D.shape == (8, 8)
    assert res.D_star.shape == (8, 8)
    assert res.f.shape == (8, 8)

    # Median across the 2D phantom recovers the ground-truth parameters within tolerance
    assert np.isclose(float(np.median(res.D)), d_true, rtol=0.08, atol=1e-4)
    assert np.isclose(float(np.median(res.D_star)), d_star_true, rtol=0.20, atol=4e-3)
    assert np.isclose(float(np.median(res.f)), f_true, rtol=0.15, atol=2.5e-2)


def test_fit_ivim_result_interfaces_and_unpacking():
    """Verify attribute access, dict lookup, tuple unpacking, and NIfTI-like image inputs."""
    phantom = generate_synthetic_phantom(
        D=0.001,
        D_star=0.02,
        f=0.15,
        snr=80.0,
        noise_type="gaussian",
        seed=7,
    )
    res = fit_ivim(phantom)
    d_hat, ds_hat, f_hat = res

    assert np.isclose(d_hat, res["D"])
    assert np.isclose(ds_hat, res["D_star"])
    assert np.isclose(f_hat, res["f"])
    assert np.isclose(d_hat, 0.001, rtol=0.08, atol=1e-4)
    assert np.isclose(ds_hat, 0.02, rtol=0.20, atol=4e-3)
    assert np.isclose(f_hat, 0.15, rtol=0.15, atol=2.5e-2)


def test_fit_ivim_nifti_like_and_mask_support():
    """Verify fit_ivim() accepts NIfTI-like image objects and spatial ROI masks."""
    class DummyNifti:
        def __init__(self, data: np.ndarray, affine: np.ndarray):
            self._data = data
            self.affine = affine

        def get_fdata(self) -> np.ndarray:
            return self._data

    phantom = generate_synthetic_phantom(
        D=0.001,
        D_star=0.02,
        f=0.15,
        snr=None,
        shape=(4, 4),
    )
    vol_4d = phantom.signal[:, :, None, :]  # (4, 4, 1, N_b)
    mask_3d = np.zeros((4, 4, 1), dtype=bool)
    mask_3d[1:3, 1:3, 0] = True

    nifti_obj = DummyNifti(vol_4d, np.eye(4))
    res = fit_ivim(phantom.bvals, nifti_obj, mask=mask_3d)

    assert res.D.shape == (4, 4, 1)
    assert np.isnan(res.D[0, 0, 0])
    assert np.isclose(res.D[1, 1, 0], 0.001, rtol=0.02, atol=5e-5)
    assert np.isclose(res.D_star[1, 1, 0], 0.02, rtol=0.05, atol=1e-3)
    assert np.isclose(res.f[1, 1, 0], 0.15, rtol=0.05, atol=1e-2)


def test_classify_railing_flags_bound_hits():
    """Verify boundary-railing classification flags lower and upper D* bound hits."""
    dstar = np.array([3e-3, 0.02, 0.15], dtype=float)
    rail = classify_railing(dstar, bounds_D_star=(3e-3, 0.15), rel_tol=1e-2)
    assert list(rail["lower_mask"]) == [True, False, False]
    assert list(rail["upper_mask"]) == [False, False, True]
    assert list(rail["railed_mask"]) == [True, False, True]
