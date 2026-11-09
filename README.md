# `sextant-core` (`sextantIVIM`) — Assumption-Free IVIM Boundary-Railing Diagnostics, NLLS Fitting, CRLB & Calibration Toolkit (`v1.0.0`)

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Typed: PEP 561](https://img.shields.io/badge/typing-PEP%20561-informational.svg)](src/sextant_core/py.typed)

`sextant-core` (`v1.0.0`) is a PEP 621 / PEP 561 strictly typed, OSIPI-ready open-source Python library for quantitative **Intravoxel Incoherent Motion (IVIM)** diffusion-weighted MRI (DWI) analysis. It provides:

1. **Assumption-Free Boundary-Railing Diagnostics** (`classify_railing`, `bootstrap_railing_ci`, `bootstrap_fraction`): Per-voxel detection and bootstrap confidence intervals for pseudo-diffusion ($D^{*}$) boundary-railing where bounded optimizers collide with physiological box constraints under weak $D^{*}$ identifiability—requiring no ground truth and no parametric noise-model assumptions.
2. **Segmented & Simultaneous Bounded NLLS Fitting** (`fit_ivim`, `fit_segmented_voxel`, `fit_nlls_voxel`, `IVIMFitResult`): Consensus two-step segmented IVIM fitting and simultaneous bounded Trust-Region Reflective non-linear least squares (NLLS) across 1D single-voxel signals, 1D voxel batches, 2D slices, 3D volumes, and 4D NIfTI images.
3. **Analytical Jacobians & Cramér–Rao Lower Bounds** (`compute_jacobian`, `compute_crlb`): Closed-form partial derivatives with respect to $(D, D^{*}, f)$ and Fisher Information Matrix / CRLB identifiability diagnostics.
4. **Rician & Gaussian Synthetic Phantoms** (`generate_synthetic_phantom`, `ivim_forward_model`, `triexp_forward_model`, `kurtosis_forward_model`): Deterministic 1D/2D multi-zone synthetic bi-exponential, tri-exponential, and diffusional-kurtosis decay generators with configurable Rician or Gaussian noise.
5. **Calibration Rulers & Contextual Stratification** (`evaluate_calibration`, `empirical_coverage`, `expected_calibration_error`, `relative_sharpness`, `summarize_df_context`): Scoped ground-truth-only Gaussian interval coverage, Expected Calibration Error (ECE), relative sharpness, and $(D, f)$ stratification across railed versus interior voxels.

> **Companion Manuscript & Reproducibility Archive:** This standalone package accompanies the *Sextant* study in [`akarlin3/researchProj/tree/main/Sextant`](https://github.com/akarlin3/researchProj/tree/main/Sextant) (*"Boundary-railing of conventional NLLS fits as an assumption-free pseudo-diffusion identifiability diagnostic in IVIM MRI"*).

---

## Installation

Install directly from GitHub via `pip`:

```bash
pip install git+https://github.com/akarlin3/sextantIVIM.git
```

Or clone the repository and install in editable mode with development, testing, and linting dependencies:

```bash
git clone https://github.com/akarlin3/sextantIVIM.git
cd sextantIVIM
pip install -e ".[dev]"
```

---

## Mathematical Formulation

### 1. Bi-Exponential IVIM Signal Decay Model

In intravoxel incoherent motion (IVIM) diffusion MRI, the diffusion-weighted signal attenuation at sensitizing gradient factor $b \ge 0$ ($\text{s/mm}^2$) is governed by a two-compartment dynamical decay law separating slow parenchymal tissue diffusion ($D$) from fast microvascular pseudo-diffusion ($D^{*} > D$):

$$
S(b) = S_0 \left[ f \exp(-b D^{*}) + (1 - f) \exp(-b D) \right]
$$

where:
- $S_0 > 0$ is the baseline unweighted ($b = 0$) signal intensity,
- $D \in [D_{\min}, D_{\max}]$ is the slow parenchymal tissue diffusivity ($\text{mm}^2/\text{s}$),
- $D^{*} \in [D^{*}_{\min}, D^{*}_{\max}]$ is the fast microvascular pseudo-diffusion coefficient ($\text{mm}^2/\text{s}$),
- $f \in [0, 1]$ is the microvascular perfusion signal fraction.

The forward model satisfies exact physical conservation laws and continuum reduction invariants:
- **Zero-Gradient Conservation**: $S(0) = S_0 [f + (1 - f)] = S_0$ for all $f \in [0, 1]$ and $(D, D^{*})$.
- **Pure Parenchymal Limit ($f \to 0$)**: $S(b; D, D^{*}, f=0) = S_0 \exp(-b D)$, independent of $D^{*}$.
- **Pure Microvascular Limit ($f \to 1$)**: $S(b; D, D^{*}, f=1) = S_0 \exp(-b D^{*})$, independent of $D$.
- **Equal-Diffusivity Degeneracy ($D^{*} \to D$)**: $S(b; D, D, f) = S_0 \exp(-b D)$ for all $f \in [0, 1]$.

### 2. Two-Step Segmented and Simultaneous Bounded NLLS Estimation

- **Consensus Two-Step Segmented Fit (`method="segmented"`)**:
  For $b \ge b_{\text{thresh}}$ (typically $200\text{ s/mm}^2$), microvascular pseudo-diffusion has decayed ($\exp(-b D^{*}) \approx 0$). Step 1 estimates $D$ and the intercept $S_{\text{int}}$ via log-linear least squares on high-$b$ signals:

$$
\ln S(b) = \ln S_{\text{int}} - b D \quad (b \ge b_{\text{thresh}}), \qquad \hat{f} = \operatorname{clip}\!\left(1 - \frac{S_{\text{int}}}{S(0)}, f_{\min}, f_{\max}\right)
$$

  Step 2 holds $(\hat{D}, \hat{f})$ fixed and optimizes $D^{*}$ over all $b$-values via bounded non-linear least squares subject to $\max(D^{*}_{\min}, \hat{D}) \le D^{*} \le D^{*}_{\max}$.

- **Simultaneous Bounded NLLS (`method="nlls"`)**:
  Warm-starts from the two-step segmented estimate and jointly minimizes the non-linear least-squares objective over $\theta = (D, D^{*}, f)$ subject to physiological box constraints:

$$
\hat{\theta} = \arg\min_{\theta \in [\theta_{\min}, \theta_{\max}]} \frac{1}{2} \sum_{i=1}^{N_b} \left( \frac{S(b_i; \theta)}{S_0} - y_i \right)^2
$$

### 3. Boundary-Railing Identifiability Diagnostic & Fisher Information

The analytical Jacobian sensitivity with respect to $D^{*}$ is:

$$
\frac{\partial S(b)}{\partial D^{*}} = -b \, S_0 \, f \exp(-b D^{*})
$$

When $f$ is small, $D^{*}$ is large, or low-$b$ sampling is sparse, $\partial S / \partial D^{*} \to 0$ and the Fisher Information Matrix $\mathcal{I}(\theta) = \sigma^{-2} J^\top J$ becomes nearly singular along the $D^{*}$ axis. A bounded trust-region solver slides along the flat objective valley until it collides with a box constraint. A voxel is classified as **boundary-railed** if its converged estimate $\hat{D}^{*}$ terminates within relative tolerance $\varepsilon_{\text{rel}}$ of either bound:

$$
\min\left( |\hat{D}^{*} - D^{*}_{\min}|, \; |\hat{D}^{*} - D^{*}_{\max}| \right) \le \varepsilon_{\text{rel}} \left( D^{*}_{\max} - D^{*}_{\min} \right)
$$

---

## Quickstart

```python
import numpy as np
from sextant_core import (
    GLOBAL_SEED,
    bootstrap_railing_ci,
    classify_railing,
    compute_jacobian,
    compute_crlb,
    evaluate_calibration,
    fit_ivim,
    generate_synthetic_phantom,
    make_rng,
)

# 1. Generate a 2D synthetic phantom (8x8 slice) with Rician noise at SNR=60
phantom = generate_synthetic_phantom(
    D=0.001,
    D_star=0.02,
    f=0.15,
    S0=100.0,
    snr=60.0,
    noise_type="rician",
    shape=(8, 8),
    seed=GLOBAL_SEED,
)

# 2. Fit using two-step segmented IVIM (or method="nlls" for simultaneous bounded NLLS)
result = fit_ivim(phantom.bvals, phantom.signal, method="segmented", b_thresh=200.0)
print(f"Median D:      {np.median(result.D):.5f} mm^2/s")
print(f"Median D*:     {np.median(result.D_star):.4f} mm^2/s")
print(f"Median f:      {np.median(result.f):.3f}")
print(f"Railed Voxels: {result.frac_railed * 100:.1f}%")

# 3. Classify boundary-railing and compute a 5,000-resample 95% bootstrap CI
railing = classify_railing(result.D_star, bounds_D_star=(3e-3, 0.05), rel_tol=1e-2)
pct, ci_lo, ci_hi = bootstrap_railing_ci(railing["railed_mask"].ravel(), n_boot=5000, seed=GLOBAL_SEED)
print(f"Railing %:     {pct:.1f}% (95% CI [{ci_lo:.1f}%, {ci_hi:.1f}%])")

# 4. Evaluate analytical Jacobian and Cramér–Rao Lower Bound (CRLB) at SNR=50
jac = compute_jacobian(phantom.bvals, D=0.001, D_star=0.02, f=0.15)
crlb = compute_crlb(phantom.bvals, D=0.001, D_star=0.02, f=0.15, snr=50.0)
print(f"Jacobian shape: {jac.shape}, CRLB D* CV: {crlb['cv_D_star']:.3f}")

# 5. Evaluate Gaussian interval calibration on synthetic ground truth
rng = make_rng(GLOBAL_SEED)
estimates = 0.02 + rng.normal(0.0, 0.003, size=5000)
cal = evaluate_calibration(estimates, truth=0.02, sigma=0.003)
print(f"ECE: {cal.ece:.4f}, 95% Coverage: {cal.coverage_by_level[0.95]:.3f}")
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
