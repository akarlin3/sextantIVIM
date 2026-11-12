"""Deterministic unit tests for the AveryCorp Suite I Stage 1 AuditReport contract in sextantIVIM."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import time

import pytest

import sextant_core
from sextant_core import AuditReport, run_audit
from sextant_core.cli import main as cli_main

REQUIRED_SCHEMA_KEYS = {
    "package",
    "version",
    "suite",
    "stage_role",
    "status",
    "verdict",
    "metrics",
    "thresholds",
    "seed",
}


def test_run_audit_contract_schema_immutability_and_determinism():
    """Verify run_audit(seed=42) returns a frozen, bit-for-bit deterministic AuditReport with exact schema."""
    t0 = time.perf_counter()
    report_a = run_audit(seed=42)
    elapsed = time.perf_counter() - t0
    report_b = run_audit(seed=42)

    assert elapsed < 2.0
    assert isinstance(report_a, AuditReport)
    assert report_a == report_b
    assert report_a.to_json() == report_b.to_json()

    with pytest.raises(FrozenInstanceError):
        report_a.status = "HALT"  # type: ignore[misc]

    as_dict = report_a.to_dict()
    assert set(as_dict.keys()) == REQUIRED_SCHEMA_KEYS

    parsed = json.loads(report_a.to_json(indent=2))
    assert set(parsed.keys()) == REQUIRED_SCHEMA_KEYS

    assert report_a.package == "sextant-ivim"
    assert report_a.version == sextant_core.__version__
    assert report_a.suite == "Suite I: Clinical Quantitative-MRI Trust Spine"
    assert (
        report_a.stage_role
        == "Stage 1: Forward NLLS Fitting, Fisher CRLB & Assumption-Free D* Boundary-Railing Gate"
    )
    assert report_a.seed == 42
    assert type(report_a.seed) is int

    assert report_a.status == "PASS"
    assert report_a.verdict.startswith("INTERIOR_IDENTIFIABLE")
    assert "INTERIOR_IDENTIFIABLE" in report_a.verdict

    for req_metric in (
        "railing_fraction_1d",
        "railing_fraction_2d",
        "median_crlb_cv_D",
        "railing_reduction",
        "verdict_code",
    ):
        assert req_metric in report_a.metrics

    for k, v in report_a.metrics.items():
        assert isinstance(k, str)
        assert type(v) in (float, int, str, bool), f"Non-native metric type for {k}: {type(v)}"

    for req_thresh in ("max_railing_fraction_2d", "max_median_crlb_cv_D"):
        assert req_thresh in report_a.thresholds

    for k, v in report_a.thresholds.items():
        assert isinstance(k, str)
        assert type(v) in (float, int, str), f"Non-native threshold type for {k}: {type(v)}"

    assert 0.0 <= float(report_a.metrics["railing_fraction_2d"]) <= float(
        report_a.thresholds["max_railing_fraction_2d"]
    )
    assert float(report_a.metrics["railing_fraction_2d"]) < float(
        report_a.metrics["railing_fraction_1d"]
    )
    assert 0.0 < float(report_a.metrics["median_crlb_cv_D"]) <= float(
        report_a.thresholds["max_median_crlb_cv_D"]
    )

    line = report_a.summary_line()
    assert "\n" not in line
    assert "sextant-ivim" in line
    assert "[PASS]" in line


def test_run_audit_collapsed_low_snr_triggers_boundary_railed_wall():
    """Verify collapsed low SNR (snr=3.0) triggers HALT and BOUNDARY_RAILED_WALL verdict."""
    low_snr_report = run_audit(seed=42, n_voxels=24, snr=3.0)
    assert isinstance(low_snr_report, AuditReport)
    assert low_snr_report.status in ("WARN", "HALT")
    assert low_snr_report.status == "HALT"
    assert low_snr_report.verdict.startswith("BOUNDARY_RAILED_WALL")
    assert "BOUNDARY_RAILED_WALL" in low_snr_report.verdict
    assert low_snr_report.metrics["verdict_code"] == "BOUNDARY_RAILED_WALL"
    assert low_snr_report.metrics["interior_identifiable"] is False

    parsed = json.loads(low_snr_report.to_json())
    assert set(parsed.keys()) == REQUIRED_SCHEMA_KEYS
    assert parsed["status"] == "HALT"

    with pytest.raises(ValueError):
        run_audit(seed=42, n_voxels=0)
    with pytest.raises(ValueError):
        run_audit(seed=42, snr=-5.0)


def test_cli_audit_and_demo_json_contracts(capsys: pytest.CaptureFixture[str]):
    """Verify CLI `sextant-ivim audit` (JSON + text) and backwards-compatible `demo --json`."""
    rc_audit_json = cli_main(["audit", "--seed", "42", "--snr", "65.0", "--n-voxels", "32", "--json"])
    assert rc_audit_json == 0
    audit_payload = json.loads(capsys.readouterr().out)
    assert set(audit_payload.keys()) == REQUIRED_SCHEMA_KEYS
    assert audit_payload["package"] == "sextant-ivim"
    assert audit_payload["suite"] == "Suite I: Clinical Quantitative-MRI Trust Spine"
    assert audit_payload["status"] == "PASS"
    assert audit_payload["verdict"].startswith("INTERIOR_IDENTIFIABLE")
    assert "railing_fraction_1d" in audit_payload["metrics"]
    assert "railing_fraction_2d" in audit_payload["metrics"]
    assert "median_crlb_cv_D" in audit_payload["metrics"]

    rc_audit_txt = cli_main(["audit", "--seed", "42", "--snr", "65.0", "--n-voxels", "16"])
    assert rc_audit_txt == 0
    audit_txt = capsys.readouterr().out
    assert "[PASS] sextant-ivim" in audit_txt
    assert "railing_fraction_2d" in audit_txt
    assert "max_railing_fraction_2d" in audit_txt

    rc_demo_json = cli_main(["demo", "--n-trials", "24", "--snr", "65.0", "--seed", "42", "--json"])
    assert rc_demo_json == 0
    demo_payload = json.loads(capsys.readouterr().out)

    # 1. Valid standardized AuditReport schema keys
    assert REQUIRED_SCHEMA_KEYS.issubset(demo_payload.keys())
    assert demo_payload["package"] == "sextant-ivim"
    assert demo_payload["version"] == sextant_core.__version__
    assert demo_payload["suite"] == "Suite I: Clinical Quantitative-MRI Trust Spine"
    assert demo_payload["status"] == "PASS"
    assert demo_payload["verdict"].startswith("INTERIOR_IDENTIFIABLE")

    # 2. 100% backwards-compatible legacy demo keys preserved
    for legacy_key in (
        "n_trials",
        "snr",
        "seed",
        "te_1d_ms",
        "match_scan_time",
        "railing_rate_1d",
        "railing_rate_2d",
        "railing_reduction",
        "railing_relative_reduction",
        "f_apparent_1d",
        "f_bias_1d",
        "f_bias_2d",
        "d_star_rmse_1d",
        "d_star_rmse_2d",
        "crlb_d_star_std_1d",
        "crlb_d_star_std_2d",
        "crlb_2d_std",
        "crlb_2d_condition_number",
    ):
        assert legacy_key in demo_payload
