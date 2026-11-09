#!/usr/bin/env python3
"""Deterministic numerical accuracy and execution-time benchmark suite for sextant-core.

Executes standardized benchmarks across two-step segmented IVIM fitting,
simultaneous bounded Trust-Region Reflective NLLS fitting, boundary-railing
diagnostics, analytical Cramér–Rao Lower Bound (CRLB) evaluation, and
uncertainty calibration / bootstrap inference under fixed seed 20260613.

Outputs a formatted Markdown table of execution times and numerical error tolerances.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure local src/ is importable when executed directly from repo root or subrepo
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _cand in (_ROOT / "src", _ROOT, _ROOT / "Sextant" / "sextant-core" / "src", _ROOT / "Sextant" / "sextant-core"):
    if _cand.exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from sextant_core import (
    GLOBAL_SEED,
    bootstrap_fraction,
    compute_crlb,
    evaluate_calibration,
    fit_ivim,
    generate_synthetic_phantom,
    make_rng,
)


@dataclass(frozen=True)
class BenchmarkRow:
    """Single row of the numerical benchmark report."""

    task: str
    scale: str
    time_ms: float
    metric_label: str
    observed_error: float
    tolerance: float
    passed: bool


def run_benchmark_suite(seed: int = GLOBAL_SEED) -> list[BenchmarkRow]:
    """Execute all deterministic numerical benchmarks and return structured rows."""
    rows: list[BenchmarkRow] = []

    d_true = 1.0e-3
    ds_true = 2.0e-2
    f_true = 0.15

    # 1. Single-voxel Segmented Fit (Noise-Free)
    ph_clean = generate_synthetic_phantom(
        D=d_true, D_star=ds_true, f=f_true, S0=1.0, snr=None, noise_type="none"
    )
    t0 = time.perf_counter()
    for _ in range(50):
        res_seg_1d = fit_ivim(ph_clean.bvals, ph_clean.signal, method="segmented")
    dt_seg_1d = (time.perf_counter() - t0) * 1000.0 / 50.0
    err_seg_d = abs(float(res_seg_1d.D) - d_true) / d_true
    rows.append(
        BenchmarkRow(
            task="Segmented IVIM Fit (1D noise-free)",
            scale="1 voxel x 12 b",
            time_ms=dt_seg_1d,
            metric_label="Rel. Error (D)",
            observed_error=err_seg_d,
            tolerance=2.0e-2,
            passed=err_seg_d <= 2.0e-2,
        )
    )

    # 2. Single-voxel Simultaneous Bounded NLLS Fit (Noise-Free)
    t0 = time.perf_counter()
    for _ in range(25):
        res_nlls_1d = fit_ivim(ph_clean.bvals, ph_clean.signal, method="nlls")
    dt_nlls_1d = (time.perf_counter() - t0) * 1000.0 / 25.0
    err_nlls_max = max(
        abs(float(res_nlls_1d.D) - d_true) / d_true,
        abs(float(res_nlls_1d.D_star) - ds_true) / ds_true,
        abs(float(res_nlls_1d.f) - f_true) / f_true,
    )
    rows.append(
        BenchmarkRow(
            task="Simultaneous Bounded NLLS (1D noise-free)",
            scale="1 voxel x 12 b",
            time_ms=dt_nlls_1d,
            metric_label="Max Rel. Error (D, D*, f)",
            observed_error=err_nlls_max,
            tolerance=1.0e-4,
            passed=err_nlls_max <= 1.0e-4,
        )
    )

    # 3. 1D Cohort Segmented Fit (Rician SNR=60, N=256)
    ph_rician = generate_synthetic_phantom(
        D=d_true,
        D_star=ds_true,
        f=f_true,
        S0=100.0,
        snr=60.0,
        noise_type="rician",
        n_voxels=256,
        seed=seed,
    )
    t0 = time.perf_counter()
    res_seg_batch = fit_ivim(ph_rician.bvals, ph_rician.signal, method="segmented")
    dt_seg_batch = (time.perf_counter() - t0) * 1000.0
    med_err_d = abs(float(np.median(res_seg_batch.D)) - d_true) / d_true
    rows.append(
        BenchmarkRow(
            task="Segmented IVIM Batch (Rician SNR=60)",
            scale="256 voxels x 12 b",
            time_ms=dt_seg_batch,
            metric_label="Median Rel. Error (D)",
            observed_error=med_err_d,
            tolerance=5.0e-2,
            passed=med_err_d <= 5.0e-2,
        )
    )

    # 4. 1D Cohort Simultaneous NLLS Fit (Rician SNR=60, N=128)
    ph_nlls_batch = generate_synthetic_phantom(
        D=d_true,
        D_star=ds_true,
        f=f_true,
        S0=100.0,
        snr=60.0,
        noise_type="rician",
        n_voxels=128,
        seed=seed,
    )
    t0 = time.perf_counter()
    res_nlls_batch = fit_ivim(ph_nlls_batch.bvals, ph_nlls_batch.signal, method="nlls")
    dt_nlls_batch = (time.perf_counter() - t0) * 1000.0
    med_err_f = abs(float(np.median(res_nlls_batch.f)) - f_true)
    rows.append(
        BenchmarkRow(
            task="Simultaneous NLLS Batch (Rician SNR=60)",
            scale="128 voxels x 12 b",
            time_ms=dt_nlls_batch,
            metric_label="Median Abs. Error (f)",
            observed_error=med_err_f,
            tolerance=2.5e-2,
            passed=med_err_f <= 2.5e-2,
        )
    )

    # 5. 2D Multi-Zone Spatial Phantom Segmented Fit (16x16, Gaussian SNR=80)
    ph_spatial = generate_synthetic_phantom(
        D=d_true,
        D_star=ds_true,
        f=f_true,
        S0=100.0,
        snr=80.0,
        noise_type="gaussian",
        shape=(16, 16),
        spatial=True,
        seed=seed,
    )
    t0 = time.perf_counter()
    res_spatial = fit_ivim(ph_spatial.bvals, ph_spatial.signal, method="segmented")
    dt_spatial = (time.perf_counter() - t0) * 1000.0
    rel_mae_spatial_d = float(
        np.median(np.abs(np.asarray(res_spatial.D) - np.asarray(ph_spatial.D)) / np.asarray(ph_spatial.D))
    )
    rows.append(
        BenchmarkRow(
            task="2D Multi-Zone Spatial Phantom (SNR=80)",
            scale="16x16 slice x 12 b",
            time_ms=dt_spatial,
            metric_label="Median Voxel Rel. Error (D)",
            observed_error=rel_mae_spatial_d,
            tolerance=6.0e-2,
            passed=rel_mae_spatial_d <= 6.0e-2,
        )
    )

    # 6. Analytical Jacobian & CRLB Fisher Matrix Evaluation (1,000 calls)
    t0 = time.perf_counter()
    for _ in range(1000):
        crlb_25 = compute_crlb(ph_clean.bvals, D=d_true, D_star=ds_true, f=f_true, snr=25.0, reg=0.0)
        crlb_50 = compute_crlb(ph_clean.bvals, D=d_true, D_star=ds_true, f=f_true, snr=50.0, reg=0.0)
    dt_crlb = (time.perf_counter() - t0) * 1000.0
    scaling_err = float(
        np.max(np.abs(np.asarray(crlb_25["std"]) / np.asarray(crlb_50["std"]) - 2.0))
    )
    rows.append(
        BenchmarkRow(
            task="Analytical Jacobian & CRLB Fisher Matrix",
            scale="1,000 evaluations",
            time_ms=dt_crlb,
            metric_label="Scaling Invariant |ratio - 2|",
            observed_error=scaling_err,
            tolerance=1.0e-12,
            passed=scaling_err <= 1.0e-12,
        )
    )

    # 7. Calibration Ruler & 5,000-Resample Bootstrap CI
    rng = make_rng(seed)
    est = ds_true + rng.normal(0.0, 0.003, size=5000)
    railed_flags = rng.random(1618) < 0.547
    t0 = time.perf_counter()
    cal = evaluate_calibration(est, ds_true, 0.003)
    _ = bootstrap_fraction(railed_flags, n_boot=5000, seed=seed)
    dt_cal = (time.perf_counter() - t0) * 1000.0
    rows.append(
        BenchmarkRow(
            task="Calibration Ruler + 5,000-Bootstrap CI",
            scale="5,000 vox / 5,000 boot",
            time_ms=dt_cal,
            metric_label="Expected Calibration Error (ECE)",
            observed_error=cal.ece,
            tolerance=1.5e-2,
            passed=cal.ece <= 1.5e-2,
        )
    )

    return rows


def format_markdown_table(rows: list[BenchmarkRow]) -> str:
    """Format benchmark results as a clean GitHub-Flavored Markdown table."""
    lines = [
        "| Benchmark Task | Input Scale | Execution Time (ms) | Error Metric | Observed Error | Tolerance | Status |",
        "| :--- | :--- | ---: | :--- | ---: | ---: | :---: |",
    ]
    for r in rows:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| {r.task} | `{r.scale}` | {r.time_ms:.2f} | {r.metric_label} "
            f"| `{r.observed_error:.3e}` | `<= {r.tolerance:.1e}` | **{status}** |"
        )
    return "\n".join(lines)


def main() -> int:
    rows = run_benchmark_suite(seed=GLOBAL_SEED)
    table_md = format_markdown_table(rows)
    print("# Sextant-Core Numerical Benchmark Suite (Seed 20260613)\n")
    print(table_md)
    if not all(r.passed for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
