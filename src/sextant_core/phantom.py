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
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np

DEFAULT_PHANTOM_BVALS: np.ndarray = np.array(
    [0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 400.0, 600.0, 800.0],
    dtype=np.float64,
)


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


def generate_synthetic_phantom(
    bvals: np.ndarray | Sequence[float] | None = None,
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
) -> SyntheticPhantom:
    """Generate 1D or 2D synthetic bi-exponential IVIM signal decay curves.

    Synthesizes diffusion-weighted signal attenuation according to:

    .. math::

        S(b) = S_0 \\left[ f \\exp(-b D^*) + (1 - f) \\exp(-b D) \\right]

    and corrupts the clean signal with configurable Rician or Gaussian noise
    at :math:`\\sigma = S_0 / \\text{SNR}`.

    Parameters
    ----------
    bvals : np.ndarray or Sequence[float], optional
        1D array of diffusion :math:`b`-values (:math:`\\text{s}/\\text{mm}^2`)
        with shape ``(N_b,)``. Defaults to a 12-point OSIPI-compatible schedule
        ``[0, 10, 20, 30, 50, 75, 100, 150, 200, 400, 600, 800]``.
    D : float or np.ndarray, default=0.001
        Ground-truth slow tissue diffusivity :math:`D` (:math:`\\text{mm}^2/\\text{s}`).
        May be a scalar or a 1D/2D spatial parameter array.
    D_star : float or np.ndarray, default=0.02
        Ground-truth fast pseudo-diffusion coefficient :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
        May be a scalar or a 1D/2D spatial parameter array.
    Dstar : float or np.ndarray or None, optional
        Backward-compatible alias for ``D_star``.
    f : float or np.ndarray, default=0.15
        Ground-truth perfusion fraction :math:`f \\in [0, 1]` (dimensionless).
        May be a scalar or a 1D/2D spatial parameter array.
    S0 : float or np.ndarray, default=1.0
        Baseline :math:`b = 0` signal intensity :math:`S_0`.
    snr : float or None, default=50.0
        Signal-to-noise ratio defined at :math:`b = 0` (:math:`\\sigma = S_0 / \\text{SNR}`).
        Pass ``None`` or ``np.inf`` for noise-free signals.
    noise_type : {"rician", "gaussian", "none"}, default="rician"
        Noise model applied to the clean signal:
        - ``"rician"``: magnitude of complex Gaussian channels
          :math:`\\sqrt{(S_{\\text{clean}} + \\epsilon_r)^2 + \\epsilon_i^2}`,
        - ``"gaussian"``: additive Gaussian noise :math:`S_{\\text{clean}} + \\epsilon`,
        - ``"none"``: noise-free signal.
    noise_model : {"rician", "gaussian", "none"} or None, optional
        Alias for ``noise_type``.
    shape : int or tuple of int, optional
        Spatial/batch shape of the generated phantom:
        - ``None`` (and scalar parameters): returns a single 1D decay curve of
          shape ``(N_b,)``.
        - ``int`` or ``(N,)``: returns a 2D array of shape ``(N, N_b)``.
        - ``(H, W)``: returns a 3D array of shape ``(H, W, N_b)`` representing a
          2D spatial slice.
    n_voxels : int, optional
        Convenience alias for ``shape=(n_voxels,)`` when generating a 1D batch
        of voxel decay curves.
    spatial : bool, default=False
        If ``True`` and ``shape`` is a 2D tuple ``(H, W)``, constructs a
        multi-zone 2D spatial phantom with distinct parenchymal, high-perfusion,
        and low-perfusion regions centered around ``(D, D_star, f)``.
    seed : int, np.random.Generator, or None, default=42
        Random seed or NumPy ``Generator`` for reproducible noise realizations.

    Returns
    -------
    SyntheticPhantom
        Phantom container with attributes ``bvals``, ``signal``, ``clean_signal``,
        ``D``, ``D_star``, ``f``, ``S0``, ``snr``, and ``noise_type``.

    Raises
    ------
    ValueError
        If ``snr <= 0`` or ``noise_type`` is not one of ``{"rician", "gaussian", "none"}``.
    """
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

    b = (
        DEFAULT_PHANTOM_BVALS.copy()
        if bvals is None
        else np.asarray(bvals, dtype=np.float64)
    )
    if b.ndim != 1 or b.size < 1:
        raise ValueError(f"`bvals` must be a non-empty 1D array; got shape {b.shape}.")

    if n_voxels is not None and shape is None:
        shape = (int(n_voxels),)
    elif isinstance(shape, int):
        shape = (shape,)

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
        clean = ivim_forward_model(b, d_gt, ds_gt, f_gt, s0_gt)
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
        clean = ivim_forward_model(b, d_gt, ds_gt, f_gt, s0_gt)

    if noise_key == "none" or snr is None or np.isinf(snr):
        noisy = clean.copy()
    else:
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
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
