"""Deterministic tests for sextant-core v1.2.0: AuditAttestation, Vernier, Procrustes, and canonical DRO bridge."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib.util
import json

import numpy as np
import pytest

from sextant_core import (
    AuditAttestation,
    ProcrustesProbeReport,
    VernierSchemeReport,
    compute_report_sha256,
    evaluate_calibration_aware_scheme,
    generate_synthetic_phantom,
    probe_d_misspecification_aliasing,
    run_audit,
)
from sextant_core.cli import main as cli_main
import sextant_core.phantoms as phantoms_alias


def test_audit_attestation_provenance_and_determinism():
    """Verify AuditAttestation embeds SHA-256 input/report digests and platform fingerprint."""
    rep1 = run_audit(seed=42)
    rep2 = run_audit(seed=42)
    assert rep1.attestation is not None
    assert isinstance(rep1.attestation, AuditAttestation)
    assert rep1.attestation == rep2.attestation

    att = rep1.attestation
    assert len(att.input_sha256) == 64
    assert len(att.report_sha256) == 64
    assert att.report_sha256 == compute_report_sha256(rep1.core_dict())
    assert {"python", "numpy", "scipy"}.issubset(att.platform_fingerprint.keys())

    with pytest.raises(FrozenInstanceError):
        att.input_sha256 = "0" * 64  # type: ignore[misc]

    rep_seed99 = run_audit(seed=99)
    assert rep_seed99.attestation is not None
    assert rep_seed99.attestation.input_sha256 != att.input_sha256
    assert rep_seed99.attestation.report_sha256 != att.report_sha256


def test_canonical_dro_bridge_and_standalone_fallback(monkeypatch: pytest.MonkeyPatch):
    """Verify sextant_core.phantoms supports 5 Lattice families & clinical schedules with/without ivimbench."""
    for sched_name, expected_len in (("osipi", 12), ("tcga_lihc", 4), ("acrin_6698", 4)):
        ph_opt = phantoms_alias.generate_synthetic_phantom(
            sched_name,
            n_voxels=16,
            snr=35.0,
            seed=42,
            use_canonical_dro=True,
            family="dispersion_gamma",
            organ="liver",
        )
        assert ph_opt.bvals.shape == (expected_len,)
        assert ph_opt.signal.shape == (16, expected_len)

    # Simulate environment without ivimbench installed
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert phantoms_alias.has_ivimbench_phantoms() is False
    for fam in ("biexp", "dispersion_gamma", "dispersion_lognormal", "stretched", "triexp"):
        ph_fb = generate_synthetic_phantom(
            "osipi",
            n_voxels=12,
            snr=40.0,
            seed=42,
            use_canonical_dro=True,
            family=fam,
            organ="liver",
        )
        assert ph_fb.signal.shape == (12, 12)
        assert np.all(np.isfinite(ph_fb.signal))


def test_vernier_calibration_aware_scheme_sharpness_and_conditional_gain():
    """Verify Vernier scheme achieves post-conformal sharpness gain and high-D* conditional recovery."""
    rep_a = evaluate_calibration_aware_scheme("osipi", nex_weights=None, snr=30.0, seed=42)
    rep_b = evaluate_calibration_aware_scheme("osipi", nex_weights=None, snr=30.0, seed=42)
    assert isinstance(rep_a, VernierSchemeReport)
    assert rep_a == rep_b
    assert np.isclose(sum(rep_a.nex_baseline), sum(rep_a.nex_vernier), atol=1e-9)
    assert rep_a.delta_sharp > 0.0
    assert rep_a.delta_cond > 0.0
    assert rep_a.high_dstar_cov_vernier > rep_a.high_dstar_cov_baseline
    assert rep_a.crlb_wall_invariance_gap <= 0.25


def test_procrustes_misspecification_marginal_vs_conditional_aliasing(
    capsys: pytest.CaptureFixture[str],
):
    """Verify bi-exponential misspecification preserves marginal D coverage while aliasing conditional strata."""
    rep = probe_d_misspecification_aliasing("osipi", family="dispersion_gamma", snr=40.0, seed=42)
    assert isinstance(rep, ProcrustesProbeReport)
    assert rep.marginal_preserved is True
    assert 0.84 <= rep.marginal_d_coverage <= 0.96
    assert rep.conditional_aliasing_detected is True
    assert abs(rep.high_f_d_bias) > abs(rep.low_f_d_bias)
    assert rep.conditional_coverage_spread > rep.well_specified_conditional_spread

    # CLI subcommands smoke check
    assert cli_main(["vernier-check", "--snr", "30", "--seed", "42", "--json"]) == 0
    v_json = json.loads(capsys.readouterr().out)
    assert v_json["delta_sharp"] == pytest.approx(
        evaluate_calibration_aware_scheme(snr=30.0, seed=42).delta_sharp
    )
    assert v_json["seed"] == 42

    assert cli_main(["procrustes-check", "--family", "dispersion_gamma", "--snr", "40", "--seed", "42", "--json"]) == 0
    p_json = json.loads(capsys.readouterr().out)
    assert p_json["marginal_preserved"] is True
    assert p_json["conditional_aliasing_detected"] is True
