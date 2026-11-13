"""Canonical phantom alias module re-exporting ``sextant_core.phantom`` and optional ``ivimbench`` bridge."""
from __future__ import annotations

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

__all__ = [
    "DEFAULT_PHANTOM_BVALS",
    "SyntheticPhantom",
    "generate_synthetic_phantom",
    "has_ivimbench_phantoms",
    "ivim_forward_model",
    "kurtosis_forward_model",
    "resolve_clinical_bvals",
    "triexp_forward_model",
]
