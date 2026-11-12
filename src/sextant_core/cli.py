"""Command-line interface (``sextant-ivim``) for IVIM boundary-railing diagnostics and volumetric fitting.

Subcommands:
- ``sextant-ivim demo``: run deterministic 1D vs 2D ``(b, TE)`` boundary-railing
  and CRLB diagnostics.
- ``sextant-ivim fit-volume``: fit 1D ``S(b)`` or 2D ``S(b, TE)`` IVIM parameter
  maps across NIfTI volumes in parallel.

License: AGPL-3.0-or-later
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import numpy as np

from . import __version__
from .calibration import GLOBAL_SEED
from .t2_ivim import compare_1d_vs_2d_railing
from .volume import fit_volume_files


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays and mappings into JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``sextant-ivim`` ``argparse.ArgumentParser``."""
    parser = argparse.ArgumentParser(
        prog="sextant-ivim",
        description=(
            "Sextant IVIM (v" + __version__ + "): assumption-free boundary-railing "
            "diagnostics, 2D (b, TE) relaxation-compensated IVIM, and multi-core NIfTI fitting."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"sextant-ivim {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_p = subparsers.add_parser(
        "demo",
        help="Run deterministic 1D vs 2D (b, TE) boundary-railing and CRLB diagnostics.",
    )
    demo_p.add_argument(
        "--n-trials",
        type=int,
        default=64,
        help="Number of Monte Carlo realizations (default: 64).",
    )
    demo_p.add_argument(
        "--snr",
        type=float,
        default=65.0,
        help="Baseline thermal SNR at (b=0, TE=0) (default: 65.0).",
    )
    demo_p.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
        help=f"PCG64 deterministic seed (default: {GLOBAL_SEED}).",
    )
    demo_p.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit machine-readable JSON diagnostics.",
    )

    vol_p = subparsers.add_parser(
        "fit-volume",
        help="Fit 1D or 2D (b, TE) IVIM parameter maps on a 4D NIfTI volume.",
    )
    vol_p.add_argument("--dwi", required=True, help="Input 4D DWI NIfTI path (.nii or .nii.gz).")
    vol_p.add_argument("--bval", required=True, help="Input diffusion b-value schedule (.bval).")
    vol_p.add_argument("--mask", default=None, help="Optional 3D brain/organ binary mask (.nii/.nii.gz).")
    vol_p.add_argument("--te", default=None, help="Optional echo-time schedule in ms (.te) for 2D fitting.")
    vol_p.add_argument(
        "--mode",
        choices=("auto", "1d", "2d"),
        default="auto",
        help="Fitting mode: 'auto' (2d if --te is given else 1d), '1d', or '2d'.",
    )
    vol_p.add_argument(
        "--method",
        choices=("segmented", "nlls"),
        default="segmented",
        help="1D IVIM fitting method when mode is 1d (default: segmented).",
    )
    vol_p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    vol_p.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Voxels per worker chunk (default: 256).",
    )
    vol_p.add_argument("--out", required=True, help="Output directory for NIfTI maps and summary.json.")
    vol_p.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit JSON summary to stdout.",
    )

    return parser


def _run_demo(args: argparse.Namespace) -> int:
    cmp_res = compare_1d_vs_2d_railing(
        n_trials=args.n_trials,
        snr=args.snr,
        seed=args.seed,
    )
    payload = {
        "package": "sextant-core",
        "version": __version__,
        "n_trials": cmp_res.n_trials,
        "snr": cmp_res.snr,
        "seed": cmp_res.seed,
        "te_1d_ms": cmp_res.te_1d,
        "match_scan_time": cmp_res.match_scan_time,
        "railing_rate_1d": cmp_res.railing_rate_1d,
        "railing_rate_2d": cmp_res.railing_rate_2d,
        "railing_reduction": cmp_res.railing_reduction,
        "railing_relative_reduction": cmp_res.railing_relative_reduction,
        "f_apparent_1d": cmp_res.f_apparent_1d,
        "f_bias_1d": cmp_res.f_bias_1d,
        "f_bias_2d": cmp_res.f_bias_2d,
        "d_star_rmse_1d": cmp_res.d_star_rmse_1d,
        "d_star_rmse_2d": cmp_res.d_star_rmse_2d,
        "crlb_d_star_std_1d": cmp_res.crlb_d_star_std_1d,
        "crlb_d_star_std_2d": cmp_res.crlb_d_star_std_2d,
        "crlb_2d_std": cmp_res.crlb_2d["crlb_std"],
        "crlb_2d_condition_number": cmp_res.crlb_2d["condition_number"],
    }
    if args.as_json:
        print(json.dumps(_to_jsonable(payload), indent=2))
        return 0

    print(f"sextant-ivim v{__version__} — 1D vs 2D (b, TE) Boundary-Railing & CRLB Demo")
    print("=" * 74)
    print(
        f"Trials: {cmp_res.n_trials} | Baseline SNR: {cmp_res.snr:.1f} | "
        f"PCG64 Seed: {cmp_res.seed} | Scan-time matched: {cmp_res.match_scan_time}"
    )
    print("-" * 74)
    print(f"1D single-TE ({cmp_res.te_1d:.0f} ms) D* railing rate : {100.0 * cmp_res.railing_rate_1d:6.2f}%")
    print(f"2D multi-echo (b, TE)  D* railing rate : {100.0 * cmp_res.railing_rate_2d:6.2f}%")
    print(
        f"Railing reduction (1D -> 2D)           : {100.0 * cmp_res.railing_reduction:+6.2f} pp "
        f"({100.0 * cmp_res.railing_relative_reduction:.1f}% relative)"
    )
    print(f"f echo-time bias (1D vs 2D)            : {cmp_res.f_bias_1d:+.4f} vs {cmp_res.f_bias_2d:+.4f}")
    print(f"D* RMSE (1D vs 2D, mm^2/s)             : {cmp_res.d_star_rmse_1d:.5f} vs {cmp_res.d_star_rmse_2d:.5f}")
    print(
        f"CRLB sigma(D*) (1D vs 2D, mm^2/s)      : "
        f"{cmp_res.crlb_d_star_std_1d:.5f} vs {cmp_res.crlb_d_star_std_2d:.5f}"
    )
    print(f"5x5 2D FIM condition number            : {cmp_res.crlb_2d['condition_number']:.3e}")
    return 0


def _run_fit_volume(args: argparse.Namespace) -> int:
    out_info = fit_volume_files(
        dwi_path=args.dwi,
        bval_path=args.bval,
        out_dir=args.out,
        mask_path=args.mask,
        te_path=args.te,
        n_jobs=args.n_jobs,
        mode=args.mode,
        method=args.method,
        chunk_size=args.chunk_size,
    )
    summary = out_info["summary"]
    if args.as_json:
        print(json.dumps(_to_jsonable(summary), indent=2))
        return 0

    frac = summary["frac_railed"]
    frac_str = "n/a" if frac is None else f"{100.0 * frac:.2f}%"
    print(
        f"sextant-ivim fit-volume ({summary['mode']} / {summary['method']}): "
        f"fitted {summary['n_voxels_fitted']}/{summary['n_voxels_total']} voxels "
        f"(railed: {summary['n_railed']} [{frac_str}]) -> {args.out}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``sextant-ivim`` console script."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "demo":
        return _run_demo(args)
    if args.command == "fit-volume":
        return _run_fit_volume(args)
    parser.error(f"Unknown command: {args.command!r}")
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
