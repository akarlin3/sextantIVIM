"""Deterministic (PCG64) unit tests for 2D (b, TE) T2-IVIM, NIfTI-1 I/O, multi-core volumes, and CLI."""
from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
from pathlib import Path
try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

import nibabel as nib
import numpy as np
import pytest

import sextant_core
from sextant_core import (
    GLOBAL_SEED,
    NiftiVolume,
    RailingComparisonResult,
    T2IVIMFitResult,
    VolumeFitResult,
    compare_1d_vs_2d_railing,
    compute_t2_ivim_crlb,
    compute_t2_ivim_jacobian,
    fit_ivim,
    fit_t2_ivim,
    fit_volume,
    fit_volume_files,
    ivim_forward_model,
    read_bvals,
    read_nifti,
    read_te_values,
    t2_ivim_forward_model,
    write_bvals,
    write_nifti,
    write_te_values,
)
from sextant_core.cli import main as cli_main
import sextant_core.fitting as fitting_mod
import sextant_core.t2_ivim as t2_ivim_mod


def _pcg64(seed: int = GLOBAL_SEED) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def test_pyproject_core_dependencies_are_numpy_scipy_light():
    """Verify pyproject.toml keeps core install strictly NumPy/SciPy-light and registers CLI."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    raw_toml = pyproject_path.read_text(encoding="utf-8")
    if tomllib is not None:
        cfg = tomllib.loads(raw_toml)
        project = cfg["project"]
        assert project["version"] == "1.2.0"
        core_deps = [d.split(">=")[0].split("==")[0].strip() for d in project["dependencies"]]
        assert core_deps == ["numpy", "scipy"]
        opt = project["optional-dependencies"]
        assert any("nibabel" in d for d in opt["nifti"])
        assert any("matplotlib" in d for d in opt["vis"])
        assert any("nibabel" in d for d in opt["all"])
        assert any("matplotlib" in d for d in opt["all"])
        assert project["scripts"]["sextant-ivim"] == "sextant_core.cli:main"
    else:
        assert 'version = "1.2.0"' in raw_toml
        assert 'sextant-ivim = "sextant_core.cli:main"' in raw_toml
        dist = importlib_metadata.distribution("sextant-core")
        reqs = [r for r in (dist.requires or []) if "extra ==" not in r]
        core_deps = [d.split(">=")[0].split("==")[0].strip() for d in reqs]
        assert core_deps == ["numpy", "scipy"]


def test_t2_ivim_forward_model_identities_and_broadcasting():
    """Verify conservation laws, 1D reduction, paired vs cartesian shapes, and vectorization."""
    b = np.array([0.0, 20.0, 60.0, 150.0, 400.0, 800.0], dtype=np.float64)
    te = np.array([25.0, 50.0, 80.0, 120.0], dtype=np.float64)

    # 1. Zero-(b, TE) baseline conservation: S(0, 0) == S0 for all f in [0, 1]
    for f_val in (0.0, 0.15, 0.50, 1.0):
        s00 = t2_ivim_forward_model([0.0], [0.0], D=1.1e-3, D_star=2.5e-2, f=f_val, T2_p=45.0, T2_t=95.0, S0=250.0)
        assert s00.shape == (1,)
        assert np.isclose(s00[0], 250.0, rtol=0.0, atol=1e-12)

    # 2. Equal-relaxation identity (T2_p == T2_t == T2): S(b, TE) == S_1D(b) * exp(-TE / T2)
    t2_iso = 80.0
    sig_grid = t2_ivim_forward_model(b, te, D=1.1e-3, D_star=2.5e-2, f=0.18, T2_p=t2_iso, T2_t=t2_iso, S0=100.0)
    assert sig_grid.shape == (b.size, te.size)
    sig_1d = ivim_forward_model(b, D=1.1e-3, D_star=2.5e-2, f=0.18, S0=100.0)
    expected_grid = sig_1d[:, None] * np.exp(-te[None, :] / t2_iso)
    assert np.allclose(sig_grid, expected_grid, rtol=0.0, atol=1e-13)

    # 3. Paired 1D schedule matches flattened cartesian grid
    b_mesh, te_mesh = np.meshgrid(b, te, indexing="ij")
    sig_paired = t2_ivim_forward_model(
        b_mesh.ravel(), te_mesh.ravel(), D=1.1e-3, D_star=2.5e-2, f=0.18, T2_p=45.0, T2_t=95.0, S0=100.0
    )
    sig_cart = t2_ivim_forward_model(
        b, te, D=1.1e-3, D_star=2.5e-2, f=0.18, T2_p=45.0, T2_t=95.0, S0=100.0
    )
    assert sig_paired.shape == (b.size * te.size,)
    assert np.allclose(sig_paired, sig_cart.ravel(), rtol=0.0, atol=1e-13)

    # 4. Vectorized parameter arrays broadcast cleanly
    d_vec = np.array([[0.9e-3, 1.2e-3], [1.4e-3, 1.0e-3]])
    f_vec = np.array([[0.10, 0.20], [0.15, 0.25]])
    sig_batch = t2_ivim_forward_model(
        b, te, D=d_vec, D_star=0.025, f=f_vec, T2_p=45.0, T2_t=95.0, S0=200.0
    )
    assert sig_batch.shape == (2, 2, b.size, te.size)


@pytest.mark.parametrize(
    "t2_pair",
    [
        (45.0, 95.0),   # Venous/capillary blood shorter than parenchyma
        (160.0, 42.0),  # Oxygenated/hepatic blood longer than parenchyma
    ],
)
def test_fit_t2_ivim_noiseless_recovery_and_fallback(
    t2_pair: tuple[float, float], monkeypatch: pytest.MonkeyPatch
):
    """Verify noiseless recovery of (D, D_star, f, T2_p, T2_t, S0) for paired, grid, batch, and pure-NumPy LM."""
    t2p_true, t2t_true = t2_pair
    d_true, ds_true, f_true, s0_true = 0.0011, 0.026, 0.18, 420.0
    b = np.array([0.0, 15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 800.0], dtype=np.float64)
    te = np.array([22.0, 42.0, 70.0, 110.0], dtype=np.float64)

    sig_2d = t2_ivim_forward_model(
        b, te, D=d_true, D_star=ds_true, f=f_true, T2_p=t2p_true, T2_t=t2t_true, S0=s0_true
    )
    res_grid = fit_t2_ivim(sig_2d, b, te)
    assert isinstance(res_grid, T2IVIMFitResult)
    assert res_grid.success is True
    assert res_grid.railed_mask is False
    assert np.isclose(res_grid.D, d_true, rtol=1e-4, atol=1e-6)
    assert np.isclose(res_grid.D_star, ds_true, rtol=1e-3, atol=1e-5)
    assert np.isclose(res_grid.f, f_true, rtol=1e-3, atol=1e-4)
    assert np.isclose(res_grid.T2_p, t2p_true, rtol=5e-3, atol=0.2)
    assert np.isclose(res_grid.T2_t, t2t_true, rtol=5e-3, atol=0.2)
    assert np.isclose(res_grid.S0, s0_true, rtol=5e-3, atol=0.5)

    # Paired 1D schedule + batched (N_vox, M) input
    b_m, te_m = np.meshgrid(b, te, indexing="ij")
    batch_sig = np.stack([sig_2d.ravel(), 2.0 * sig_2d.ravel()], axis=0)
    res_batch = fit_t2_ivim(batch_sig, b_m.ravel(), te_m.ravel())
    assert res_batch.D.shape == (2,)
    assert np.allclose(res_batch.D, [d_true, d_true], rtol=1e-4)
    assert np.allclose(res_batch.S0, [s0_true, 2.0 * s0_true], rtol=5e-3)
    assert set(res_batch.railed_params.keys()) == {"D", "D_star", "f", "T2_p", "T2_t"}

    # Verify pure-NumPy damped LM fallback when scipy.optimize.least_squares is unavailable
    monkeypatch.setattr(t2_ivim_mod, "_scipy_least_squares", None)
    res_lm = fit_t2_ivim(sig_2d, b, te)
    assert np.isclose(res_lm.D, d_true, rtol=5e-3, atol=2e-5)
    assert np.isclose(res_lm.D_star, ds_true, rtol=2e-2, atol=5e-4)
    assert np.isclose(res_lm.f, f_true, rtol=2e-2, atol=3e-3)


def test_t2_ivim_crlb_matches_finite_difference_jacobian():
    """Verify exact 5x5 (and 6x6) FIM/CRLB against central finite-difference Jacobians."""
    b = np.array([0.0, 15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 800.0], dtype=np.float64)
    te = np.array([22.0, 42.0, 70.0, 110.0], dtype=np.float64)
    params = {"D": 0.0011, "D_star": 0.025, "f": 0.16, "T2_p": 45.0, "T2_t": 95.0, "S0": 2.5}

    crlb5 = compute_t2_ivim_crlb(b, te, **params, sigma=0.05)
    assert crlb5["fim"].shape == (5, 5)
    assert crlb5["cov"].shape == (5, 5)
    assert set(crlb5["crlb_std"].keys()) == {"D", "D_star", "f", "T2_p", "T2_t"}
    assert crlb5["condition_number"] > 1.0
    assert np.allclose(crlb5["fim"] @ crlb5["cov"], np.eye(5), rtol=1e-7, atol=1e-9)

    jac6 = compute_t2_ivim_jacobian(b, te, **params, include_s0=True)
    assert jac6.shape == (b.size * te.size, 6)
    assert np.allclose(jac6[:, :5], crlb5["jacobian"], rtol=0.0, atol=1e-15)

    steps = {"D": 1e-8, "D_star": 1e-7, "f": 1e-7, "T2_p": 1e-5, "T2_t": 1e-5, "S0": 1e-6}
    for col_idx, name in enumerate(("D", "D_star", "f", "T2_p", "T2_t", "S0")):
        h = steps[name]
        p_plus = dict(params)
        p_minus = dict(params)
        p_plus[name] += h
        p_minus[name] -= h
        fd_col = (
            t2_ivim_forward_model(b, te, **p_plus).ravel()
            - t2_ivim_forward_model(b, te, **p_minus).ravel()
        ) / (2.0 * h)
        assert np.allclose(jac6[:, col_idx], fd_col, rtol=1e-6, atol=1e-9)

    # Exact 1/SNR scaling of CRLB standard deviations
    crlb_snr40 = compute_t2_ivim_crlb(b, te, **params, snr=40.0)
    crlb_snr80 = compute_t2_ivim_crlb(b, te, **params, snr=80.0)
    assert np.allclose(crlb_snr40["std"] / crlb_snr80["std"], 2.0, rtol=1e-7)


def test_compare_1d_vs_2d_railing_reduces_railing_and_echo_time_bias():
    """Verify deterministic PCG64 2D (b, TE) reduces D* railing and f bias vs 1D single-TE."""
    cmp1 = compare_1d_vs_2d_railing(n_trials=64, snr=65.0, seed=GLOBAL_SEED)
    cmp2 = compare_1d_vs_2d_railing(n_trials=64, snr=65.0, seed=GLOBAL_SEED)
    assert isinstance(cmp1, RailingComparisonResult)
    assert cmp1.railing_rate_1d == cmp2.railing_rate_1d
    assert cmp1.railing_rate_2d == cmp2.railing_rate_2d
    assert cmp1.f_bias_2d == cmp2.f_bias_2d

    # 2D (b, TE) reduces D* boundary-railing, f echo-time bias, D* RMSE, and CRLB std(D*)
    assert cmp1["railing_rate_2d"] < cmp1["railing_rate_1d"]
    assert cmp1["railing_reduction"] > 0.10
    assert abs(cmp1["f_bias_2d"]) < abs(cmp1["f_bias_1d"])
    assert cmp1["d_star_rmse_2d"] < cmp1["d_star_rmse_1d"]
    assert cmp1["crlb_d_star_std_2d"] < cmp1["crlb_d_star_std_1d"]


@pytest.mark.parametrize("ext", [".nii", ".nii.gz"])
@pytest.mark.parametrize(
    "dt",
    [np.float32, np.float64, np.int16, np.uint8],
)
def test_pure_numpy_nifti_roundtrip_and_nibabel_cross_compat(
    tmp_path: Path, ext: str, dt: type[np.generic]
):
    """Verify pure-NumPy NIfTI-1 roundtrip AND bidirectional cross-compatibility with nibabel."""
    rng = _pcg64(101)
    if np.issubdtype(dt, np.floating):
        arr = rng.normal(50.0, 12.0, size=(5, 4, 3, 2)).astype(dt)
    elif dt is np.int16:
        arr = rng.integers(-500, 1500, size=(5, 4, 3, 2), dtype=np.int16)
    else:
        arr = rng.integers(0, 250, size=(5, 4, 3, 2), dtype=np.uint8)

    # Non-trivial rotated + anisotropic voxel + translated affine
    theta = 0.25
    c, s = float(np.cos(theta)), float(np.sin(theta))
    affine = np.array(
        [
            [2.0 * c, -1.8 * s, 0.0, -32.5],
            [2.0 * s, 1.8 * c, 0.0, 18.25],
            [0.0, 0.0, 3.5, -12.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # 1. Pure-NumPy write -> Pure-NumPy read AND nibabel read
    p_np = tmp_path / f"pure_np_{np.dtype(dt).name}{ext}"
    write_nifti(arr, p_np, affine=affine, dtype=dt, use_nibabel=False)
    vol_np = read_nifti(p_np, use_nibabel=False)
    assert isinstance(vol_np, NiftiVolume)
    assert vol_np.header["backend"] == "numpy"
    assert vol_np.data.dtype == np.dtype(dt)
    assert np.allclose(vol_np.data, arr, rtol=1e-6, atol=1e-6)
    assert np.allclose(vol_np.affine, affine, rtol=1e-5, atol=1e-5)

    nib_img = nib.load(str(p_np))
    assert np.allclose(np.asarray(nib_img.dataobj), arr, rtol=1e-6, atol=1e-6)
    assert np.allclose(nib_img.affine, affine, rtol=1e-5, atol=1e-5)

    # 2. nibabel write -> Pure-NumPy read
    p_nib = tmp_path / f"nib_written_{np.dtype(dt).name}{ext}"
    out_nib = nib.Nifti1Image(arr, affine)
    out_nib.header.set_data_dtype(dt)
    nib.save(out_nib, str(p_nib))

    vol_from_nib = read_nifti(p_nib, use_nibabel=False)
    assert vol_from_nib.header["backend"] == "numpy"
    assert np.allclose(vol_from_nib.data, arr, rtol=1e-6, atol=1e-6)
    assert np.allclose(vol_from_nib.affine, affine, rtol=1e-5, atol=1e-5)


def test_nifti_scaling_qform_and_fitting_fallback_without_nibabel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify scl_slope/scl_inter scaling, qform-only affine decoding, and fitting.py without nibabel."""
    rng = _pcg64(202)
    phys_data = rng.uniform(10.0, 90.0, size=(4, 3, 2)).astype(np.float64)
    p_scaled = tmp_path / "scaled_int16.nii.gz"
    write_nifti(
        phys_data,
        p_scaled,
        dtype=np.int16,
        scl_slope=0.01,
        scl_inter=10.0,
        use_nibabel=False,
    )
    vol_scaled = read_nifti(p_scaled, use_nibabel=False)
    assert np.allclose(vol_scaled.data, phys_data, atol=0.01)

    # qform-only NIfTI written by nibabel (sform_code=0, qform_code=1) -> read by pure-NumPy
    affine_q = np.diag([1.5, 2.0, 2.5, 1.0])
    affine_q[:3, 3] = [-10.0, 20.0, -5.0]
    q_img = nib.Nifti1Image(np.ones((3, 3, 2), dtype=np.float32), affine_q)
    q_img.header.set_sform(None, code=0)
    q_img.header.set_qform(affine_q, code=1)
    p_qonly = tmp_path / "qform_only.nii.gz"
    nib.save(q_img, str(p_qonly))
    vol_q = read_nifti(p_qonly, use_nibabel=False)
    assert vol_q.header["sform_code"] == 0
    assert vol_q.header["qform_code"] == 1
    assert np.allclose(vol_q.affine, affine_q, atol=1e-5)

    # Verify fitting.py loads & exports NIfTI cleanly when nibabel is absent
    bvals = np.array([0.0, 20.0, 50.0, 150.0, 400.0, 800.0], dtype=np.float64)
    dwi_4d = ivim_forward_model(
        bvals,
        D=np.full((2, 2, 1), 0.0011),
        D_star=np.full((2, 2, 1), 0.022),
        f=np.full((2, 2, 1), 0.15),
        S0=100.0,
    )
    dwi_path = tmp_path / "dwi_no_nib.nii.gz"
    write_nifti(dwi_4d.astype(np.float32), dwi_path, affine=affine_q, use_nibabel=False)

    monkeypatch.setattr(fitting_mod, "nib", None)
    fit_res = fit_ivim(bvals, dwi_path, method="segmented")
    assert fit_res.D.shape == (2, 2, 1)
    assert np.allclose(fit_res.affine, affine_q, atol=1e-5)
    assert np.isclose(fit_res.D[0, 0, 0], 0.0011, rtol=2e-2)

    nii_container = fit_res.to_nifti("D")
    assert isinstance(nii_container, NiftiVolume)
    saved_map_path = fit_res.save_nifti(tmp_path / "D_fallback.nii.gz", param="D")
    reloaded = read_nifti(saved_map_path, use_nibabel=False)
    assert np.allclose(reloaded.data, fit_res.D.astype(np.float32), atol=1e-7)


def test_fit_volume_multicore_exact_parity_1d_and_2d():
    """Verify exact bit-for-bit agreement between n_jobs=1 and n_jobs=2 on masked 4D volumes."""
    rng = _pcg64(303)
    shape_3d = (3, 3, 2)
    mask_3d = np.ones(shape_3d, dtype=bool)
    mask_3d[0, 0, 0] = False
    mask_3d[2, 2, 1] = False

    # 1. 1D volumetric fitting (n_jobs=1 vs n_jobs=2 with small chunk_size=4)
    bvals = np.array([0.0, 15.0, 30.0, 60.0, 150.0, 400.0, 800.0], dtype=np.float64)
    clean_1d = ivim_forward_model(
        bvals,
        D=rng.uniform(0.8e-3, 1.4e-3, size=shape_3d),
        D_star=rng.uniform(0.015, 0.035, size=shape_3d),
        f=rng.uniform(0.10, 0.25, size=shape_3d),
        S0=100.0,
    )
    dwi_1d = np.clip(clean_1d + rng.normal(0.0, 1.0, size=clean_1d.shape), 1e-6, None)

    v1 = fit_volume(dwi_1d, bvals, mask=mask_3d, mode="1d", method="nlls", n_jobs=1, chunk_size=4)
    v2 = fit_volume(dwi_1d, bvals, mask=mask_3d, mode="1d", method="nlls", n_jobs=2, chunk_size=4)
    assert isinstance(v1, VolumeFitResult)
    assert v1.n_voxels_fitted == int(np.sum(mask_3d))
    for key in ("D", "D_star", "f", "S0", "railed_mask"):
        assert np.array_equal(v1[key], v2[key], equal_nan=True)
    assert np.isnan(v2.D[0, 0, 0])

    # 2. 2D (b, TE) volumetric fitting (n_jobs=1 vs n_jobs=2 with small chunk_size=4)
    b_ax = np.array([0.0, 20.0, 60.0, 150.0, 400.0, 800.0], dtype=np.float64)
    te_ax = np.array([24.0, 48.0, 85.0], dtype=np.float64)
    b_m, te_m = np.meshgrid(b_ax, te_ax, indexing="ij")
    b_sched, te_sched = b_m.ravel(), te_m.ravel()
    clean_2d = t2_ivim_forward_model(
        b_sched,
        te_sched,
        D=np.full(shape_3d, 0.0011),
        D_star=np.full(shape_3d, 0.025),
        f=np.full(shape_3d, 0.16),
        T2_p=np.full(shape_3d, 45.0),
        T2_t=np.full(shape_3d, 95.0),
        S0=100.0,
    )
    dwi_2d = np.clip(clean_2d + rng.normal(0.0, 0.5, size=clean_2d.shape), 1e-6, None)

    u1 = fit_volume(
        dwi_2d, b_sched, mask=mask_3d, te_values=te_sched, mode="2d", n_jobs=1, chunk_size=4
    )
    u2 = fit_volume(
        dwi_2d, b_sched, mask=mask_3d, te_values=te_sched, mode="2d", n_jobs=2, chunk_size=4
    )
    for key in ("D", "D_star", "f", "T2_p", "T2_t", "S0", "railed_mask"):
        assert np.array_equal(u1[key], u2[key], equal_nan=True)
    assert np.isclose(float(np.nanmedian(u2.T2_p)), 45.0, rtol=0.15)
    assert np.isclose(float(np.nanmedian(u2.T2_t)), 95.0, rtol=0.10)


def test_cli_demo_and_fit_volume_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Verify sextant-ivim CLI `demo` and `fit-volume` (1D & 2D) subcommands end-to-end."""
    # 1. CLI demo (--json and human-readable)
    rc_json = cli_main(["demo", "--n-trials", "24", "--snr", "65.0", "--seed", "20260613", "--json"])
    assert rc_json == 0
    demo_out = json.loads(capsys.readouterr().out)
    assert demo_out["version"] == sextant_core.__version__
    assert demo_out["railing_rate_2d"] < demo_out["railing_rate_1d"]

    rc_txt = cli_main(["demo", "--n-trials", "16", "--seed", "20260613"])
    assert rc_txt == 0
    txt_out = capsys.readouterr().out
    assert "sextant-ivim v1.2.0" in txt_out
    assert "Railing reduction" in txt_out

    # 2. CLI fit-volume (2D multi-echo volume with mask)
    b_ax = np.array([0.0, 20.0, 60.0, 150.0, 400.0, 800.0], dtype=np.float64)
    te_ax = np.array([25.0, 55.0, 95.0], dtype=np.float64)
    b_m, te_m = np.meshgrid(b_ax, te_ax, indexing="ij")
    b_sched, te_sched = b_m.ravel(), te_m.ravel()

    dwi_4d = t2_ivim_forward_model(
        b_sched,
        te_sched,
        D=np.full((2, 2, 2), 0.0011),
        D_star=np.full((2, 2, 2), 0.024),
        f=np.full((2, 2, 2), 0.16),
        T2_p=np.full((2, 2, 2), 45.0),
        T2_t=np.full((2, 2, 2), 95.0),
        S0=200.0,
    )
    mask_3d = np.ones((2, 2, 2), dtype=np.uint8)
    mask_3d[0, 0, 0] = 0

    dwi_path = write_nifti(dwi_4d.astype(np.float32), tmp_path / "dwi2d.nii.gz", use_nibabel=False)
    mask_path = write_nifti(mask_3d, tmp_path / "mask.nii.gz", use_nibabel=False)
    bval_path = write_bvals(b_sched, tmp_path / "acq.bval")
    te_path = write_te_values(te_sched, tmp_path / "acq.te")
    assert np.allclose(read_bvals(bval_path), b_sched)
    assert np.allclose(read_te_values(te_path), te_sched)

    out_dir = tmp_path / "maps_2d"
    rc_vol = cli_main(
        [
            "fit-volume",
            "--dwi",
            str(dwi_path),
            "--bval",
            str(bval_path),
            "--mask",
            str(mask_path),
            "--te",
            str(te_path),
            "--n-jobs",
            "2",
            "--chunk-size",
            "3",
            "--out",
            str(out_dir),
            "--json",
        ]
    )
    assert rc_vol == 0
    vol_summary = json.loads(capsys.readouterr().out)
    assert vol_summary["mode"] == "2d"
    assert vol_summary["n_voxels_fitted"] == 7

    for fname in (
        "D.nii.gz",
        "D_star.nii.gz",
        "f.nii.gz",
        "T2_p.nii.gz",
        "T2_t.nii.gz",
        "S0.nii.gz",
        "railed_mask.nii.gz",
        "summary.json",
    ):
        assert (out_dir / fname).exists()

    # Also verify direct fit_volume_files on 1D schedule
    b_1d = np.array([0.0, 20.0, 50.0, 150.0, 400.0, 800.0], dtype=np.float64)
    dwi_1d = ivim_forward_model(
        b_1d, D=np.full((2, 2, 1), 0.001), D_star=np.full((2, 2, 1), 0.02), f=np.full((2, 2, 1), 0.15), S0=100.0
    )
    dwi_1d_p = write_nifti(dwi_1d.astype(np.float32), tmp_path / "dwi1d.nii.gz", use_nibabel=False)
    bval_1d_p = write_bvals(b_1d, tmp_path / "dwi1d.bval")
    out_1d = fit_volume_files(dwi_1d_p, bval_1d_p, tmp_path / "maps_1d", n_jobs=1, use_nibabel=False)
    assert out_1d["summary"]["mode"] == "1d"
    assert (tmp_path / "maps_1d" / "D.nii.gz").exists()
    assert (tmp_path / "maps_1d" / "summary.json").exists()
