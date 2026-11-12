# `sextant-core` (`sextantIVIM`) — Assumption-Free IVIM Boundary-Railing Diagnostics, 2D `(b, TE)` Relaxation-Compensated Fitting, CRLB & Calibration Toolkit (`v1.1.0`)

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Typed: PEP 561](https://img.shields.io/badge/typing-PEP%20561-informational.svg)](src/sextant_core/py.typed)

`sextant-core` (`v1.1.0`) is a PEP 621 / PEP 561 strictly typed, OSIPI-ready open-source Python library for quantitative **Intravoxel Incoherent Motion (IVIM)** and **2D `(b, TE)` diffusion–relaxation** MRI analysis, with a lightweight core (`numpy` + `scipy` only). It provides:

1. **Assumption-Free Boundary-Railing Diagnostics** (`classify_railing`, `bootstrap_railing_ci`, `bootstrap_fraction`): Per-voxel detection and bootstrap confidence intervals for pseudo-diffusion ($D^{*}$) boundary-railing where bounded optimizers collide with physiological box constraints under weak $D^{*}$ identifiability—requiring no ground truth and no parametric noise-model assumptions.
2. **Segmented & Simultaneous Bounded NLLS Fitting** (`fit_ivim`, `fit_segmented_voxel`, `fit_nlls_voxel`, `IVIMFitResult`): Consensus two-step segmented IVIM fitting and simultaneous bounded Trust-Region Reflective non-linear least squares (NLLS) across 1D single-voxel signals, 1D voxel batches, 2D slices, 3D volumes, and 4D NIfTI images.
3. **2D `(b, TE)` Relaxation-Compensated IVIM & Railing Reduction** (`t2_ivim_forward_model`, `fit_t2_ivim`, `compute_t2_ivim_crlb`, `compare_1d_vs_2d_railing`, `T2IVIMFitResult`): Joint two-compartment diffusion–relaxation modeling separating microvascular ($D^{*}, T_{2,p}$) from parenchymal ($D, T_{2,t}$) transverse relaxation, exact analytical $5 \times 5$ Fisher Information / CRLB identifiability analysis, and deterministic (`PCG64`) 1D vs 2D boundary-railing and echo-time $f$-bias benchmarks.
4. **Pure-NumPy / `nibabel` NIfTI-1 I/O & Multi-Core Volumetric CLI** (`read_nifti`, `write_nifti`, `read_bvals`, `fit_volume`, `fit_volume_files`, `sextant-ivim`): Zero-extra-dependency 348-byte NIfTI-1 (`.nii` / `.nii.gz`) reader/writer (automatically accelerating with `nibabel` when installed), chunked multi-process 3D/4D volume fitting via `ProcessPoolExecutor`, and the `sextant-ivim` command-line interface.
5. **Analytical Jacobians & Cramér–Rao Lower Bounds** (`compute_jacobian`, `compute_crlb`, `compute_t2_ivim_jacobian`, `compute_t2_ivim_crlb`): Closed-form partial derivatives and Fisher Information Matrix / CRLB identifiability diagnostics.
6. **Rician & Gaussian Synthetic Phantoms & Calibration Rulers** (`generate_synthetic_phantom`, `ivim_forward_model`, `triexp_forward_model`, `kurtosis_forward_model`, `evaluate_calibration`, `empirical_coverage`, `expected_calibration_error`, `relative_sharpness`, `summarize_df_context`): Deterministic 1D/2D synthetic decay generators and ground-truth-scoped Gaussian interval calibration metrics.

> **Companion Manuscript & Reproducibility Archive:** This standalone package accompanies the *Sextant* study in [`akarlin3/researchProj/tree/main/Sextant`](https://github.com/akarlin3/researchProj/tree/main/Sextant) (*"Boundary-railing of conventional NLLS fits as an assumption-free pseudo-diffusion identifiability diagnostic in IVIM MRI"*).

---

## Installation

Install the lightweight NumPy/SciPy core directly from GitHub via `pip`:

```bash
pip install git+https://github.com/akarlin3/sextantIVIM.git
```

Optional extras (`nifti` for `nibabel`, `vis` for `matplotlib`, `all`, and `dev`):

```bash
git clone https://github.com/akarlin3/sextantIVIM.git
cd sextantIVIM
pip install -e ".[all,dev]"
```

---

## Mathematical Formulation

### 1. Bi-Exponential 1D IVIM & 2D `(b, TE)` Diffusion–Relaxation Signal Models

In standard 1D single-echo IVIM diffusion MRI, the diffusion-weighted signal attenuation at sensitizing gradient factor $b \ge 0$ ($\text{s/mm}^2$) separates slow parenchymal tissue diffusion ($D$) from fast microvascular pseudo-diffusion ($D^{*} > D$):

$$
S(b) = S_0 \left[ f \exp(-b D^{*}) + (1 - f) \exp(-b D) \right]
$$

When microvascular blood and parenchymal tissue exhibit distinct transverse relaxation times ($T_{2,p} \ne T_{2,t}$, in $\text{ms}$), acquiring multi-echo $(b, \text{TE})$ schedules enables the **2D relaxation-compensated IVIM model**:

$$
S(b, \text{TE}) = S_0 \left[ f \exp(-b D^{*}) \exp(-\text{TE}/T_{2,p}) + (1 - f) \exp(-b D) \exp(-\text{TE}/T_{2,t}) \right]
$$

where:
- $S_0 > 0$ is the unweighted ($b = 0, \text{TE} = 0$) proton-density baseline amplitude,
- $D \in [D_{\min}, D_{\max}]$ is the slow parenchymal tissue diffusivity ($\text{mm}^2/\text{s}$),
- $D^{*} \in [D^{*}_{\min}, D^{*}_{\max}]$ is the fast microvascular pseudo-diffusion coefficient ($\text{mm}^2/\text{s}$),
- $f \in [0, 1]$ is the relaxation-compensated microvascular perfusion fraction,
- $T_{2,p}, T_{2,t} > 0$ are the microvascular and parenchymal transverse relaxation times ($\text{ms}$).

At a fixed single echo time $\text{TE}$, 1D IVIM conflates $f$ with differential $T_2$ weighting:

$$
f_{\text{app}}(\text{TE}) = \frac{f \exp(-\text{TE}/T_{2,p})}{f \exp(-\text{TE}/T_{2,p}) + (1 - f) \exp(-\text{TE}/T_{2,t})}
$$

Joint 2D $(b, \text{TE})$ estimation disentangles $(T_{2,p}, T_{2,t})$, eliminating echo-time $f$ bias and substantially reducing $D^{*}$ boundary-railing under equal scan-time budgets.

The forward models satisfy exact physical conservation laws and continuum reduction invariants:
- **Zero-Gradient Conservation**: $S(0) = S_0$ and $S(0, 0) = S_0$ for all $f \in [0, 1]$.
- **Pure Parenchymal Limit ($f \to 0$)**: $S(b, \text{TE}) = S_0 \exp(-b D - \text{TE}/T_{2,t})$, independent of $(D^{*}, T_{2,p})$.
- **Pure Microvascular Limit ($f \to 1$)**: $S(b, \text{TE}) = S_0 \exp(-b D^{*} - \text{TE}/T_{2,p})$, independent of $(D, T_{2,t})$.
- **Iso-Relaxation Reduction ($T_{2,p} = T_{2,t} = T_2$)**: $S(b, \text{TE}) = S_{\text{1D}}(b) \exp(-\text{TE}/T_2)$.

### 2. Two-Step Segmented and Simultaneous Bounded NLLS Estimation

- **Consensus Two-Step Segmented Fit (`method="segmented"`)**:
  For $b \ge b_{\text{thresh}}$ (typically $200\text{ s/mm}^2$), microvascular pseudo-diffusion has decayed ($\exp(-b D^{*}) \approx 0$). Step 1 estimates $D$ and the intercept $S_{\text{int}}$ via log-linear least squares on high-$b$ signals:

$$
\ln S(b) = \ln S_{\text{int}} - b D \quad (b \ge b_{\text{thresh}}), \qquad \hat{f} = \operatorname{clip}\!\left(1 - \frac{S_{\text{int}}}{S(0)}, f_{\min}, f_{\max}\right)
$$

  Step 2 holds $(\hat{D}, \hat{f})$ fixed and optimizes $D^{*}$ over all $b$-values via bounded non-linear least squares subject to $\max(D^{*}_{\min}, \hat{D}) \le D^{*} \le D^{*}_{\max}$.

- **Simultaneous Bounded NLLS (`method="nlls"` and `fit_t2_ivim`)**:
  Warm-starts from separable log-linear / Golub–Pereyra variable-projection candidates and jointly minimizes the bounded non-linear least-squares objective with exact analytical Jacobians and $D \le D^{*}$ identifiability enforcement.

### 3. Boundary-Railing Identifiability Diagnostic & Fisher Information

In 2D $(b, \text{TE})$ IVIM, the analytical $5 \times 5$ Fisher Information Matrix over $\theta = (D, D^{*}, f, T_{2,p}, T_{2,t})$ is $\mathcal{I}(\theta) = \sigma^{-2} J^\top J$. When $f$ is small, $D^{*}$ is large, or low-$b$ / short-$\text{TE}$ sampling is sparse, $\partial S / \partial D^{*} \to 0$ and bounded solvers slide along the flat valley until colliding with a box constraint:

$$
\min\left( |\hat{D}^{*} - D^{*}_{\min}|, \; |\hat{D}^{*} - D^{*}_{\max}| \right) \le \varepsilon_{\text{rel}} \left( D^{*}_{\max} - D^{*}_{\min} \right)
$$

---

## Quickstart

### 1D IVIM Fitting, Boundary-Railing & Calibration

```python
import numpy as np
from sextant_core import (
    GLOBAL_SEED,
    bootstrap_railing_ci,
    classify_railing,
    compute_crlb,
    compute_jacobian,
    evaluate_calibration,
    fit_ivim,
    generate_synthetic_phantom,
    make_rng,
)

# 1. Generate a 2D synthetic phantom (8x8 slice) with Rician noise at SNR=60
phantom = generate_synthetic_phantom(
    D=0.001, D_star=0.02, f=0.15, S0=100.0, snr=60.0, noise_type="rician", shape=(8, 8), seed=GLOBAL_SEED
)

# 2. Fit using two-step segmented IVIM (or method="nlls" for simultaneous bounded NLLS)
result = fit_ivim(phantom.bvals, phantom.signal, method="segmented", b_thresh=200.0)
print(f"Median D: {np.median(result.D):.5f} mm^2/s | Railed: {result.frac_railed * 100:.1f}%")

# 3. Bootstrap 95% CI on boundary-railing percentage
railing = classify_railing(result.D_star, bounds_D_star=(3e-3, 0.05), rel_tol=1e-2)
pct, ci_lo, ci_hi = bootstrap_railing_ci(railing["railed_mask"].ravel(), n_boot=5000, seed=GLOBAL_SEED)

# 4. Analytical Jacobian, CRLB, and Gaussian calibration ruler
jac = compute_jacobian(phantom.bvals, D=0.001, D_star=0.02, f=0.15)
crlb = compute_crlb(phantom.bvals, D=0.001, D_star=0.02, f=0.15, snr=50.0)
rng = make_rng(GLOBAL_SEED)
cal = evaluate_calibration(0.02 + rng.normal(0.0, 0.003, size=5000), truth=0.02, sigma=0.003)
```

### 2D `(b, TE)` Relaxation-Compensated IVIM, 5x5 CRLB & Multi-Core Volumes (`v1.1.0`)

```python
import numpy as np
from sextant_core import (
    compare_1d_vs_2d_railing,
    compute_t2_ivim_crlb,
    fit_t2_ivim,
    fit_volume,
    t2_ivim_forward_model,
)

bvals = np.array([0.0, 15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 800.0])
te_vals = np.array([22.0, 42.0, 70.0, 110.0])

# Forward model & bounded 2D NLLS recovery
sig_2d = t2_ivim_forward_model(bvals, te_vals, D=0.0011, D_star=0.025, f=0.16, T2_p=45.0, T2_t=95.0, S0=250.0)
fit2d = fit_t2_ivim(sig_2d, bvals, te_vals)
print(fit2d.D, fit2d.D_star, fit2d.f, fit2d.T2_p, fit2d.T2_t, fit2d.railed_mask)

# Exact 5x5 Fisher Information Matrix & CRLB over (D, D*, f, T2_p, T2_t)
crlb2d = compute_t2_ivim_crlb(bvals, te_vals, D=0.0011, D_star=0.025, f=0.16, T2_p=45.0, T2_t=95.0, snr=65.0)
print("5x5 CRLB std:", crlb2d["crlb_std"], "cond:", crlb2d["condition_number"])

# Deterministic PCG64 Monte Carlo comparison: 1D single-TE vs 2D (b, TE) under equal scan time
cmp = compare_1d_vs_2d_railing(n_trials=128, snr=65.0)
print(f"1D railing: {cmp.railing_rate_1d:.1%} -> 2D railing: {cmp.railing_rate_2d:.1%}")
print(f"f bias 1D: {cmp.f_bias_1d:+.4f} -> 2D: {cmp.f_bias_2d:+.4f}")

# Multi-core 4D volumetric fitting with ProcessPoolExecutor
vol_4d = np.broadcast_to(sig_2d.ravel(), (4, 4, 2, bvals.size * te_vals.size)).copy()
b_grid, te_grid = np.meshgrid(bvals, te_vals, indexing="ij")
maps = fit_volume(vol_4d, b_grid.ravel(), te_values=te_grid.ravel(), mode="2d", n_jobs=2)
```

### Command-Line Interface (`sextant-ivim`)

```bash
# Run deterministic 1D vs 2D (b, TE) boundary-railing & 5x5 CRLB diagnostic demo
sextant-ivim demo --n-trials 64 --snr 65.0 --seed 20260613
sextant-ivim demo --n-trials 64 --json

# Fit 1D or 2D (b, TE) NIfTI parameter maps across CPU cores (works with or without nibabel)
sextant-ivim fit-volume \
  --dwi dwi_4d.nii.gz \
  --bval dwi.bval \
  --mask brain_mask.nii.gz \
  --te dwi.te \
  --n-jobs 4 \
  --out ivim_maps/
```

---

## Deterministic Numerical Benchmarks

Reproduce the 7-task benchmark suite (fixed seed `20260613`) by running:

```bash
python benchmarks/run_benchmarks.py
```

| Benchmark Task (`python benchmarks/run_benchmarks.py`) | Input Scale | Execution Time (ms) | Error Metric | Observed Error | Tolerance | Status |
| :--- | :--- | ---: | :--- | ---: | ---: | :---: |
| Segmented IVIM Fit (1D noise-free) | `1 voxel x 12 b` | 1.27 | Rel. Error (D) | `5.953e-03` | `<= 2.0e-02` | **PASS** |
| Simultaneous Bounded NLLS (1D noise-free) | `1 voxel x 12 b` | 2.26 | Max Rel. Error (D, D*, f) | `2.428e-12` | `<= 1.0e-04` | **PASS** |
| Segmented IVIM Batch (Rician SNR=60) | `256 voxels x 12 b` | 341.16 | Median Rel. Error (D) | `1.348e-02` | `<= 5.0e-02` | **PASS** |
| Simultaneous NLLS Batch (Rician SNR=60) | `128 voxels x 12 b` | 395.29 | Median Abs. Error (f) | `3.356e-03` | `<= 2.5e-02` | **PASS** |
| 2D Multi-Zone Spatial Phantom (SNR=80) | `16x16 slice x 12 b` | 328.09 | Median Voxel Rel. Error (D) | `4.478e-02` | `<= 6.0e-02` | **PASS** |
| Analytical Jacobian & CRLB Fisher Matrix | `1,000 evaluations` | 40.44 | Scaling Invariant \|ratio - 2\| | `0.000e+00` | `<= 1.0e-12` | **PASS** |
| Calibration Ruler + 5,000-Bootstrap CI | `5,000 vox / 5,000 boot` | 36.84 | Expected Calibration Error (ECE) | `7.760e-03` | `<= 1.5e-02` | **PASS** |

---

## Companion Manuscript & Citation

If you use `sextant-core` in your research, please cite the software package (see [`CITATION.cff`](CITATION.cff)) and the companion manuscript in [`akarlin3/researchProj/tree/main/Sextant`](https://github.com/akarlin3/researchProj/tree/main/Sextant):

- Karlin, A. (2026). *Boundary-railing of conventional NLLS fits as an assumption-free pseudo-diffusion identifiability diagnostic in IVIM MRI*. **NMR in Biomedicine**.
- Karlin, A. (2026). *Distribution-Free Conformal Coverage for IVIM Parameter Maps, and the Identifiability Wall in the Pseudo-Diffusion Compartment*. **Magnetic Resonance in Medicine**.

---

## License

This repository is licensed under the **GNU Affero General Public License v3.0** (`AGPL-3.0`); see [`LICENSE`](LICENSE) for the full license text.
