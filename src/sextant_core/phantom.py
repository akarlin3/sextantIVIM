"""Synthetic bi-exponential IVIM-DWI phantom generator.

Generates 1D (single-voxel or multi-voxel) and 2D spatial bi-exponential
IVIM signal decay curves according to:

.. math::

    S(b) = S_0 \\left[ f \\exp(-b D^*) + (1 - f) \\exp(-b D) \\right]

with configurable Rician or Gaussian noise at a specified signal-to-noise
ratio (:math:`\\text{SNR} = S_0 / \\sigma`).
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np

DEFAULT_PHANTOM_BVALS: np.ndarray = np.array(
    [0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 400.0, 600.0, 800.0],
    dtype=np.float64,
)

_FALLBACK_CLINICAL_SCHEDULES: dict[str, tuple[float, ...]] = {
    "osipi": (0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 400.0, 600.0, 800.0),
    "tcga_lihc": (0.0, 50.0, 400.0, 800.0),
    "acrin_6698": (0.0, 100.0, 600.0, 800.0),
    "lattice_22b": (
        0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0,
        90.0, 110.0, 130.0, 150.0, 175.0, 200.0, 250.0, 350.0, 500.0, 650.0, 800.0,
    ),
}


def has_ivimbench_phantoms() -> bool:
    """Return ``True`` when ``ivimbench.phantoms`` is importable at runtime."""
    return importlib.util.find_spec("ivimbench") is not None


def resolve_clinical_bvals(
    bvals: np.ndarray | Sequence[float] | str | None = None,
    *,
    use_canonical_dro: bool = False,
) -> np.ndarray:
    """Resolve a 1D ``b``-value array or named clinical schedule (``osipi``, ``tcga_lihc``, ``acrin_6698``)."""
    if bvals is None:
        if use_canonical_dro and has_ivimbench_phantoms():
            import importlib as _importlib

            proto_mod = _importlib.import_module("ivimbench.protocols")
            res = proto_mod.resolve_bvalues("osipi")
            arr = res[0] if isinstance(res, tuple) else res
            return np.asarray(arr, dtype=np.float64)
        return DEFAULT_PHANTOM_BVALS.copy()
    if isinstance(bvals, str):
        key = bvals.strip().lower()
        if use_canonical_dro and has_ivimbench_phantoms():
            import importlib as _importlib

            proto_mod = _importlib.import_module("ivimbench.protocols")
            res = proto_mod.resolve_bvalues(key)
            arr = res[0] if isinstance(res, tuple) else res
            return np.asarray(arr, dtype=np.float64)
        if key in _FALLBACK_CLINICAL_SCHEDULES:
            return np.asarray(_FALLBACK_CLINICAL_SCHEDULES[key], dtype=np.float64)
        raise ValueError(
            f"Unknown clinical b-value schedule {bvals!r}; supported fallback schedules: "
            f"{sorted(_FALLBACK_CLINICAL_SCHEDULES)}."
        )
    return np.asarray(bvals, dtype=np.float64)


@dataclass
class SyntheticPhantom(Mapping[str, Any]):
    """Container for synthetic IVIM phantom signals and ground-truth parameters.

    Supports attribute access (``phantom.bvals``, ``phantom.signal``,
    ``phantom.D``, ``phantom.D_star``, ``phantom.f``), dictionary-style lookup
    (``phantom["signal"]``, ``phantom["D_star"]``), and 2-tuple unpacking
    (``bvals, signal = generate_synthetic_phantom(...)``).

    Parameters
    ----------
    bvals : np.ndarray
        1D array of diffusion :math:`b`-values (:math:`\\text{s}/\\text{mm}^2`)
        with shape ``(N_b,)``.
    signal : np.ndarray
        Simulated (noisy) DWI signal array with shape ``(N_b,)`` for a single
        1D decay curve, ``(N_voxels, N_b)`` for a 1D batch of curves, or
        ``(H, W, N_b)`` for a 2D spatial phantom.
    clean_signal : np.ndarray
        Noise-free bi-exponential signal array with the same shape as ``signal``.
    D : np.ndarray or float
        Ground-truth tissue diffusivity :math:`D` (:math:`\\text{mm}^2/\\text{s}`).
    D_star : np.ndarray or float
        Ground-truth pseudo-diffusion coefficient :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
    f : np.ndarray or float
        Ground-truth perfusion fraction :math:`f \\in [0, 1]` (dimensionless).
    S0 : np.ndarray or float
        Ground-truth baseline :math:`b = 0` signal intensity :math:`S_0`.
    snr : float or None
        Configured signal-to-noise ratio (:math:`S_0 / \\sigma`), or ``None``
        for noise-free signals.
    noise_type : {"rician", "gaussian", "none"}
        Noise distribution applied to ``clean_signal``.
    """

    bvals: np.ndarray
    signal: np.ndarray
    clean_signal: np.ndarray
    D: np.ndarray | float
    D_star: np.ndarray | float
    f: np.ndarray | float
    S0: np.ndarray | float
    snr: float | None = 50.0
    noise_type: str = "rician"

    @property
    def Dstar(self) -> np.ndarray | float:
        """Alias for :attr:`D_star`."""
        return self.D_star

    @property
    def dstar(self) -> np.ndarray | float:
        """Lowercase alias for :attr:`D_star`."""
        return self.D_star

    @property
    def signals(self) -> np.ndarray:
        """Alias for :attr:`signal`."""
        return self.signal

    @property
    def ground_truth(self) -> dict[str, np.ndarray | float]:
        """Dictionary of ground-truth parameter maps ``{"D", "D_star", "f", "S0"}``."""
        return {
            "D": self.D,
            "D_star": self.D_star,
            "f": self.f,
            "S0": self.S0,
        }

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield ``(bvals, signal)`` for convenient 2-tuple unpacking."""
        yield self.bvals
        yield self.signal

    def __len__(self) -> int:
        return 9

    def keys(self) -> list[str]:  # type: ignore[override]
        return [
            "bvals",
            "signal",
            "clean_signal",
            "D",
            "D_star",
            "f",
            "S0",
            "snr",
            "noise_type",
        ]

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            pair = (self.bvals, self.signal)
            return pair[key]
        mapping = {
            "bvals": self.bvals,
            "signal": self.signal,
            "signals": self.signal,
            "clean_signal": self.clean_signal,
            "D": self.D,
            "D_star": self.D_star,
            "Dstar": self.D_star,
            "dstar": self.D_star,
            "f": self.f,
            "S0": self.S0,
            "snr": self.snr,
            "noise_type": self.noise_type,
            "ground_truth": self.ground_truth,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(f"Unknown key {key!r} in SyntheticPhantom")


def ivim_forward_model(
    bvals: np.ndarray | Sequence[float],
    D: np.ndarray | float,
    D_star: np.ndarray | float,
    f: np.ndarray | float,
    S0: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Evaluate the clean bi-exponential IVIM signal decay model.

    Computes:

    .. math::

        S(b) = S_0 \\left[ f \\exp(-b D^*) + (1 - f) \\exp(-b D) \\right]

    Parameters
    ----------
    bvals : np.ndarray or Sequence[float]
        1D array of diffusion :math:`b`-values (:math:`\\text{s}/\\text{mm}^2`)
        with shape ``(N_b,)``.
    D : np.ndarray or float
        Slow tissue diffusivity :math:`D` (:math:`\\text{mm}^2/\\text{s}`).
    D_star : np.ndarray or float
        Fast pseudo-diffusion coefficient :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
    f : np.ndarray or float
        Perfusion fraction :math:`f \\in [0, 1]` (dimensionless).
    S0 : np.ndarray or float, default=1.0
        Baseline :math:`b = 0` signal intensity :math:`S_0`.

    Returns
    -------
    np.ndarray
        Clean IVIM signal array of shape ``(*param_shape, N_b)``.
    """
    b = np.asarray(bvals, dtype=np.float64)
    d_arr = np.asarray(D, dtype=np.float64)[..., None]
    ds_arr = np.asarray(D_star, dtype=np.float64)[..., None]
    f_arr = np.asarray(f, dtype=np.float64)[..., None]
    s0_arr = np.asarray(S0, dtype=np.float64)[..., None]

    return s0_arr * (f_arr * np.exp(-b * ds_arr) + (1.0 - f_arr) * np.exp(-b * d_arr))


def triexp_forward_model(
    bvals: np.ndarray | Sequence[float],
    D: np.ndarray | float,
    D_star: np.ndarray | float,
    f: np.ndarray | float,
    S0: np.ndarray | float = 1.0,
    w: float = 0.25,
    slow: float = 0.35,
) -> np.ndarray:
    """Evaluate a three-compartment (tri-exponential) diffusion signal decay model.

    Models a tissue compartment carrying a slow restricted sub-compartment of
    weight ``w`` and diffusivity ``slow * D``. Reduces identically to
    :func:`ivim_forward_model` when ``w == 0.0`` or ``slow == 1.0``.

    Parameters
    ----------
    bvals : np.ndarray or Sequence[float]
        1D array of diffusion :math:`b`-values (:math:`\\text{s}/\\text{mm}^2`).
    D : np.ndarray or float
        Primary parenchymal tissue diffusivity :math:`D` (:math:`\\text{mm}^2/\\text{s}`).
    D_star : np.ndarray or float
        Fast pseudo-diffusion coefficient :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
    f : np.ndarray or float
        Perfusion fraction :math:`f \\in [0, 1]`.
    S0 : np.ndarray or float, default=1.0
        Baseline :math:`b = 0` signal intensity :math:`S_0`.
    w : float, default=0.25
        Relative weight in :math:`[0, 1]` of the slow restricted tissue tail.
    slow : float, default=0.35
        Ratio of slow restricted diffusivity to primary tissue diffusivity :math:`D`.

    Returns
    -------
    np.ndarray
        Clean tri-exponential signal array of shape ``(*param_shape, N_b)``.
    """
    b = np.asarray(bvals, dtype=np.float64)
    d_arr = np.asarray(D, dtype=np.float64)[..., None]
    ds_arr = np.asarray(D_star, dtype=np.float64)[..., None]
    f_arr = np.asarray(f, dtype=np.float64)[..., None]
    s0_arr = np.asarray(S0, dtype=np.float64)[..., None]
    tissue = (1.0 - float(w)) * np.exp(-b * d_arr) + float(w) * np.exp(-b * float(slow) * d_arr)
    return s0_arr * (f_arr * np.exp(-b * ds_arr) + (1.0 - f_arr) * tissue)


def kurtosis_forward_model(
    bvals: np.ndarray | Sequence[float],
    D: np.ndarray | float,
    D_star: np.ndarray | float,
    f: np.ndarray | float,
    S0: np.ndarray | float = 1.0,
    K: float = 0.8,
) -> np.ndarray:
    """Evaluate a non-Gaussian diffusional-kurtosis IVIM signal decay model.

    Models non-Gaussian parenchymal hindrance via apparent diffusional kurtosis
    :math:`K` in the tissue compartment:

    .. math::

        S(b) = S_0 \\left[ f \\exp(-b D^*) + (1 - f) \\exp\\left(-b D + \\frac{1}{6}(b D)^2 K\\right) \\right]

    Reduces identically to :func:`ivim_forward_model` when ``K == 0.0``.

    Parameters
    ----------
    bvals : np.ndarray or Sequence[float]
        1D array of diffusion :math:`b`-values (:math:`\\text{s}/\\text{mm}^2`).
    D : np.ndarray or float
        Slow tissue diffusivity :math:`D` (:math:`\\text{mm}^2/\\text{s}`).
    D_star : np.ndarray or float
        Fast pseudo-diffusion coefficient :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
    f : np.ndarray or float
        Perfusion fraction :math:`f \\in [0, 1]`.
    S0 : np.ndarray or float, default=1.0
        Baseline :math:`b = 0` signal intensity :math:`S_0`.
    K : float, default=0.8
        Dimensionless apparent diffusional kurtosis coefficient.

    Returns
    -------
    np.ndarray
        Clean kurtosis-IVIM signal array of shape ``(*param_shape, N_b)``.
    """
    b = np.asarray(bvals, dtype=np.float64)
    d_arr = np.asarray(D, dtype=np.float64)[..., None]
    ds_arr = np.asarray(D_star, dtype=np.float64)[..., None]
    f_arr = np.asarray(f, dtype=np.float64)[..., None]
    s0_arr = np.asarray(S0, dtype=np.float64)[..., None]
    bd = b * d_arr
    tissue = np.exp(-bd + (1.0 / 6.0) * (bd**2) * float(K))
    return s0_arr * (f_arr * np.exp(-b * ds_arr) + (1.0 - f_arr) * tissue)


def _build_spatial_2d_maps(
    shape: tuple[int, int],
    base_D: float,
    base_D_star: float,
    base_f: float,
    base_S0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a multi-region 2D anatomical phantom around the base parameters."""
    h, w = shape
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float64),
        np.linspace(-1.0, 1.0, w, dtype=np.float64),
        indexing="ij",
    )
    radius = np.sqrt(xx**2 + yy**2)

    # Background parenchyma centered on (base_D, base_D_star, base_f)
    D_map = np.full(shape, base_D, dtype=np.float64)
    D_star_map = np.full(shape, base_D_star, dtype=np.float64)
    f_map = np.full(shape, base_f, dtype=np.float64)
    S0_map = np.full(shape, base_S0, dtype=np.float64)

    # Inner high-perfusion region
    inner = radius < 0.45
    D_map[inner] = np.clip(base_D * 1.25, 2e-4, 4e-3)
    D_star_map[inner] = np.clip(base_D_star * 1.35, 5e-3, 0.12)
    f_map[inner] = np.clip(base_f * 1.6, 0.02, 0.45)

    # Outer rim with restricted diffusion / lower perfusion
    outer = radius > 0.85
    D_map[outer] = np.clip(base_D * 0.75, 2e-4, 4e-3)
    D_star_map[outer] = np.clip(base_D_star * 0.75, 5e-3, 0.12)
    f_map[outer] = np.clip(base_f * 0.65, 0.02, 0.45)

    return D_map, D_star_map, f_map, S0_map


def _evaluate_lattice_family_signal(
    b: np.ndarray,
    d_gt: np.ndarray | float,
    ds_gt: np.ndarray | float,
    f_gt: np.ndarray | float,
    s0_gt: np.ndarray | float,
    *,
    family: str = "biexp",
    use_canonical_dro: bool = False,
) -> np.ndarray:
    """Evaluate 5-family Lattice IVIM signal via ``ivimbench.phantoms`` if requested and available, else standalone NumPy."""
    fam = str(family).strip().lower()
    if use_canonical_dro and has_ivimbench_phantoms():
        import importlib as _importlib

        ph_mod = _importlib.import_module("ivimbench.phantoms")
        d_flat = np.atleast_1d(np.asarray(d_gt, dtype=np.float64)).ravel()
        ds_flat = np.atleast_1d(np.asarray(ds_gt, dtype=np.float64)).ravel()
        f_flat = np.atleast_1d(np.asarray(f_gt, dtype=np.float64)).ravel()
        s0_flat = np.atleast_1d(np.asarray(s0_gt, dtype=np.float64)).ravel()
        if fam == "biexp":
            flat_sig = ph_mod.ivim_biexp(b, d_flat, ds_flat, f_flat, s0=s0_flat)
        elif fam == "dispersion_gamma":
            flat_sig = ph_mod.ivim_dispersion_gamma(b, d_flat, ds_flat, 4.0, f_flat, s0=s0_flat)
        elif fam == "dispersion_lognormal":
            flat_sig = ph_mod.ivim_dispersion_lognormal(b, d_flat, ds_flat, 0.5, f_flat, s0=s0_flat)
        elif fam == "stretched":
            flat_sig = ph_mod.ivim_stretched(b, d_flat, ds_flat, f_flat, 0.7, s0=s0_flat)
        elif fam == "triexp":
            flat_sig = ph_mod.ivim_triexp(b, d_flat, ds_flat, f_flat, 4.0 * ds_flat, 0.3, s0=s0_flat)
        elif fam == "kurtosis":
            flat_sig = ph_mod.ivim_kurtosis(b, d_flat, ds_flat, f_flat, K=0.75, s0=s0_flat)
        else:
            raise ValueError(f"Unsupported phantom family {family!r}.")
        param_shape = np.asarray(d_gt).shape
        return np.asarray(flat_sig, dtype=np.float64).reshape((*param_shape, b.size))

    # Standalone pure-NumPy 5-family fallback
    if fam == "biexp":
        return ivim_forward_model(b, d_gt, ds_gt, f_gt, s0_gt)
    if fam == "dispersion_gamma":
        d_arr = np.asarray(d_gt, dtype=np.float64)[..., None]
        ds_arr = np.asarray(ds_gt, dtype=np.float64)[..., None]
        f_arr = np.asarray(f_gt, dtype=np.float64)[..., None]
        s0_arr = np.asarray(s0_gt, dtype=np.float64)[..., None]
        k_val = 4.0
        perf = np.power(1.0 + b * ds_arr / k_val, -k_val)
        return s0_arr * (f_arr * perf + (1.0 - f_arr) * np.exp(-b * d_arr))
    if fam == "dispersion_lognormal":
        d_arr, ds_arr, f_arr, s0_arr = np.broadcast_arrays(
            np.asarray(d_gt, dtype=np.float64),
            np.asarray(ds_gt, dtype=np.float64),
            np.asarray(f_gt, dtype=np.float64),
            np.asarray(s0_gt, dtype=np.float64),
        )
        orig_shape = d_arr.shape
        d_c = d_arr.reshape(-1, 1)
        ds_c = ds_arr.reshape(-1, 1)
        f_c = f_arr.reshape(-1, 1)
        s0_c = s0_arr.reshape(-1, 1)
        cv_val = 0.5
        sig_ln = np.sqrt(np.log1p(cv_val**2))
        m_ln = np.log(np.maximum(ds_c, 1e-12)) - 0.5 * sig_ln**2
        nodes, weights = np.polynomial.hermite_e.hermegauss(32)
        w = (weights / np.sqrt(2.0 * np.pi))[None, None, :]
        z = nodes[None, None, :]
        ds_grid = np.exp(m_ln[..., None] + sig_ln * z)
        perf = np.sum(w * np.exp(-b[None, :, None] * ds_grid), axis=-1)
        out = s0_c * (f_c * perf + (1.0 - f_c) * np.exp(-b[None, :] * d_c))
        return out.reshape((*orig_shape, b.size))
    if fam == "stretched":
        d_arr = np.asarray(d_gt, dtype=np.float64)[..., None]
        ds_arr = np.asarray(ds_gt, dtype=np.float64)[..., None]
        f_arr = np.asarray(f_gt, dtype=np.float64)[..., None]
        s0_arr = np.asarray(s0_gt, dtype=np.float64)[..., None]
        perf = np.exp(-np.power(np.clip(b * ds_arr, 0.0, None), 0.7))
        return s0_arr * (f_arr * perf + (1.0 - f_arr) * np.exp(-b * d_arr))
    if fam == "triexp":
        d_arr = np.asarray(d_gt, dtype=np.float64)[..., None]
        ds_arr = np.asarray(ds_gt, dtype=np.float64)[..., None]
        f_arr = np.asarray(f_gt, dtype=np.float64)[..., None]
        s0_arr = np.asarray(s0_gt, dtype=np.float64)[..., None]
        g_val = 0.3
        return s0_arr * (
            (1.0 - f_arr) * np.exp(-b * d_arr)
            + f_arr * (1.0 - g_val) * np.exp(-b * ds_arr)
            + f_arr * g_val * np.exp(-b * (4.0 * ds_arr))
        )
    if fam == "kurtosis":
        return kurtosis_forward_model(b, d_gt, ds_gt, f_gt, S0=s0_gt, K=0.75)
    raise ValueError(
        f"Unsupported phantom family {family!r}; expected one of "
        "('biexp', 'dispersion_gamma', 'dispersion_lognormal', 'stretched', 'triexp', 'kurtosis')."
    )


def generate_synthetic_phantom(
    bvals: np.ndarray | Sequence[float] | str | None = None,
    *,
    D: float | np.ndarray = 0.001,
    D_star: float | np.ndarray = 0.02,
    Dstar: float | np.ndarray | None = None,
    f: float | np.ndarray = 0.15,
    S0: float | np.ndarray = 1.0,
    snr: float | None = 50.0,
    noise_type: Literal["rician", "gaussian", "none"] = "rician",
    noise_model: Literal["rician", "gaussian", "none"] | None = None,
    shape: int | tuple[int, ...] | None = None,
    n_voxels: int | None = None,
    spatial: bool = False,
    seed: int | np.random.Generator | None = 42,
    use_canonical_dro: bool = False,
    family: str = "biexp",
    organ: str | None = None,
) -> SyntheticPhantom:
    """Generate 1D or 2D synthetic IVIM signal decay curves with optional canonical DRO delegation."""
    if Dstar is not None:
        D_star = Dstar
    if noise_model is not None:
        noise_type = noise_model

    noise_key = str(noise_type).lower()
    if noise_key not in ("rician", "gaussian", "none"):
        raise ValueError(
            f"Unsupported noise_type {noise_type!r}; expected 'rician', 'gaussian', or 'none'."
        )
    if snr is not None and not np.isinf(snr) and float(snr) <= 0:
        raise ValueError(f"`snr` must be positive or None/inf; got {snr}.")

    b = resolve_clinical_bvals(bvals, use_canonical_dro=use_canonical_dro)
    if b.ndim != 1 or b.size < 1:
        raise ValueError(f"`bvals` must be a non-empty 1D array; got shape {b.shape}.")

    if n_voxels is not None and shape is None:
        shape = (int(n_voxels),)
    elif isinstance(shape, int):
        shape = (shape,)

    # Optional canonical IVIMBench organ-prior / 5-family Lattice cohort bridge
    if use_canonical_dro and organ is not None and shape is not None and len(shape) == 1:
        n_vox_req = int(shape[0])
        seed_int = 42 if isinstance(seed, np.random.Generator) or seed is None else int(seed)
        if has_ivimbench_phantoms():
            import importlib as _importlib

            ph_mod = _importlib.import_module("ivimbench.phantoms")
            eff_snr = 1e9 if (noise_key == "none" or snr is None or np.isinf(snr)) else float(snr)
            eff_noise = "gaussian" if noise_key == "none" else noise_key
            cohort = ph_mod.generate_cohort(
                protocol=b,
                organ=organ,
                n_voxels=n_vox_req,
                snr=eff_snr,
                family=family,
                noise_type=eff_noise,
                seed=seed_int,
            )
            clean_c = np.asarray(cohort.signals_clean, dtype=np.float64) * float(np.asarray(S0).flat[0])
            noisy_c = (
                clean_c.copy()
                if (noise_key == "none" or snr is None or np.isinf(snr))
                else np.asarray(cohort.signals, dtype=np.float64) * float(np.asarray(S0).flat[0])
            )
            s0_vec = np.full(n_vox_req, float(np.asarray(S0).flat[0]), dtype=np.float64)
            return SyntheticPhantom(
                bvals=b,
                signal=noisy_c,
                clean_signal=clean_c,
                D=np.asarray(cohort.D, dtype=np.float64),
                D_star=np.asarray(cohort.Dstar, dtype=np.float64),
                f=np.asarray(cohort.f, dtype=np.float64),
                S0=s0_vec,
                snr=(None if (snr is None or np.isinf(snr)) else float(snr)),
                noise_type=noise_key,
            )
        # Standalone PCG64 fallback when ivimbench is absent
        rng_prior = np.random.Generator(np.random.PCG64(seed_int))
        D = rng_prior.uniform(0.6e-3, 1.8e-3, size=n_vox_req)
        D_star = np.exp(rng_prior.uniform(np.log(15e-3), np.log(90e-3), size=n_vox_req))
        f = rng_prior.uniform(0.08, 0.35, size=n_vox_req)

    d_in = np.asarray(D, dtype=np.float64)
    ds_in = np.asarray(D_star, dtype=np.float64)
    f_in = np.asarray(f, dtype=np.float64)
    s0_in = np.asarray(S0, dtype=np.float64)

    all_scalar = (
        d_in.ndim == 0
        and ds_in.ndim == 0
        and f_in.ndim == 0
        and s0_in.ndim == 0
    )

    if shape is None and all_scalar:
        d_gt: np.ndarray | float = float(d_in)
        ds_gt: np.ndarray | float = float(ds_in)
        f_gt: np.ndarray | float = float(f_in)
        s0_gt: np.ndarray | float = float(s0_in)
        clean = _evaluate_lattice_family_signal(
            b, d_gt, ds_gt, f_gt, s0_gt, family=family, use_canonical_dro=use_canonical_dro
        )
    else:
        if shape is not None and len(shape) == 2 and spatial and all_scalar:
            d_gt, ds_gt, f_gt, s0_gt = _build_spatial_2d_maps(
                shape=(int(shape[0]), int(shape[1])),
                base_D=float(d_in),
                base_D_star=float(ds_in),
                base_f=float(f_in),
                base_S0=float(s0_in),
            )
        else:
            target_shape = (
                tuple(int(s) for s in shape)
                if shape is not None
                else np.broadcast_shapes(d_in.shape, ds_in.shape, f_in.shape, s0_in.shape)
            )
            d_gt = np.broadcast_to(d_in, target_shape).astype(np.float64).copy()
            ds_gt = np.broadcast_to(ds_in, target_shape).astype(np.float64).copy()
            f_gt = np.broadcast_to(f_in, target_shape).astype(np.float64).copy()
            s0_gt = np.broadcast_to(s0_in, target_shape).astype(np.float64).copy()
        clean = _evaluate_lattice_family_signal(
            b, d_gt, ds_gt, f_gt, s0_gt, family=family, use_canonical_dro=use_canonical_dro
        )

    if noise_key == "none" or snr is None or np.isinf(snr):
        noisy = clean.copy()
    else:
        rng = (
            seed
            if isinstance(seed, np.random.Generator)
            else np.random.Generator(np.random.PCG64(42 if seed is None else int(seed)))
        )
        sigma = np.asarray(s0_gt, dtype=np.float64)[..., None] / float(snr)
        if noise_key == "rician":
            re = clean + rng.normal(0.0, sigma, size=clean.shape)
            im = rng.normal(0.0, sigma, size=clean.shape)
            noisy = np.sqrt(re**2 + im**2)
        else:
            eps = rng.normal(0.0, sigma, size=clean.shape)
            noisy = np.clip(clean + eps, 1e-12, None)

    return SyntheticPhantom(
        bvals=b,
        signal=noisy,
        clean_signal=clean,
        D=d_gt,
        D_star=ds_gt,
        f=f_gt,
        S0=s0_gt,
        snr=(None if (snr is None or np.isinf(snr)) else float(snr)),
        noise_type=noise_key,
    )
