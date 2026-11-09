"""Core bi-exponential IVIM-DWI fitting and boundary-railing diagnostics.

Provides OSIPI-compliant two-step segmented and simultaneous bounded
non-linear least-squares (NLLS) fitting for intravoxel incoherent motion (IVIM)
diffusion-weighted imaging (DWI), fully decoupled from local filesystem paths.

The forward signal model is the standard bi-exponential decay:

.. math::

    S(b) = S_0 \\left[ f \\exp(-b D^*) + (1 - f) \\exp(-b D) \\right]

where:
- :math:`S_0` is the baseline (:math:`b = 0`) signal intensity,
- :math:`D` is the slow parenchymal tissue diffusivity (:math:`\\text{mm}^2/\\text{s}`),
- :math:`D^*` is the fast microvascular pseudo-diffusion coefficient (:math:`\\text{mm}^2/\\text{s}`),
- :math:`f \\in [0, 1]` is the microvascular perfusion fraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np

try:
    from scipy.optimize import least_squares as _scipy_least_squares
except ImportError:  # pragma: no cover - fallback when scipy is absent
    _scipy_least_squares = None

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - optional runtime import
    nib = None


# Default OSIPI 10-point abdominal evaluation b-value schedule (s/mm^2)
DEFAULT_BVALS: np.ndarray = np.array(
    [0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 400.0, 600.0],
    dtype=np.float64,
)

# Default physiological parameter bounds: (lower, upper)
DEFAULT_BOUNDS_D: tuple[float, float] = (1e-4, 5e-3)
DEFAULT_BOUNDS_D_STAR: tuple[float, float] = (3e-3, 0.15)
DEFAULT_BOUNDS_F: tuple[float, float] = (0.0, 0.7)
DEFAULT_B_THRESH: float = 200.0
DEFAULT_RAIL_TOL: float = 1e-2


def _bounded_lm_1d(
    bvals: np.ndarray,
    y: np.ndarray,
    tail: np.ndarray,
    f_hat: float,
    x0: float,
    lo: float,
    hi: float,
    max_iter: int = 80,
    tol: float = 1e-9,
) -> float:
    """Pure-NumPy bounded Levenberg-Marquardt solver for 1D D* optimization."""
    if _scipy_least_squares is not None:
        def residuals(p: np.ndarray) -> np.ndarray:
            return (tail + f_hat * np.exp(-bvals * p[0])) - y

        res = _scipy_least_squares(
            residuals,
            x0=np.array([x0], dtype=np.float64),
            bounds=([lo], [hi]),
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=200,
        )
        return float(res.x[0])

    # Fallback: multistart grid + projected Levenberg-Marquardt
    grid = np.linspace(lo, hi, 25, dtype=np.float64)
    preds = tail[None, :] + f_hat * np.exp(-grid[:, None] * bvals[None, :])
    losses = np.sum((preds - y[None, :]) ** 2, axis=1)
    x = float(grid[int(np.argmin(losses))])
    if abs(x0 - np.clip(x0, lo, hi)) < 1e-12:
        r0 = (tail + f_hat * np.exp(-bvals * x0)) - y
        if float(np.sum(r0 * r0)) < float(np.min(losses)):
            x = float(x0)

    lam = 1e-4
    for _ in range(max_iter):
        exp_term = np.exp(-bvals * x)
        r = (tail + f_hat * exp_term) - y
        cost = 0.5 * float(np.dot(r, r))
        jac = -bvals * f_hat * exp_term
        grad = float(np.dot(jac, r))
        hess = float(np.dot(jac, jac))
        if abs(grad) < tol:
            break
        step = -grad / (hess + lam + 1e-16)
        x_new = float(np.clip(x + step, lo, hi))
        r_new = (tail + f_hat * np.exp(-bvals * x_new)) - y
        cost_new = 0.5 * float(np.dot(r_new, r_new))
        if cost_new < cost:
            if abs(x_new - x) < tol:
                x = x_new
                break
            x = x_new
            lam = max(lam * 0.3, 1e-12)
        else:
            lam *= 5.0
            if lam > 1e10:
                break
    return x


def _bounded_lm_3d(
    bvals: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    s0: float = 1.0,
    max_iter: int = 120,
    tol: float = 1e-9,
) -> np.ndarray:
    """Bounded NLLS solver for simultaneous 3-parameter (D, D_star, f) IVIM fit."""
    if _scipy_least_squares is not None:
        def residuals(p: np.ndarray) -> np.ndarray:
            d, ds, f = p[0], p[1], p[2]
            pred = s0 * (f * np.exp(-bvals * ds) + (1.0 - f) * np.exp(-bvals * d))
            return pred - y

        try:
            res = _scipy_least_squares(
                residuals,
                x0=np.clip(x0, lo + 1e-12, hi - 1e-12),
                bounds=(lo, hi),
                method="trf",
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=400,
            )
            return res.x.astype(np.float64)
        except Exception:
            return np.full(3, np.nan, dtype=np.float64)

    # Pure-NumPy projected Levenberg-Marquardt fallback
    x = np.clip(np.asarray(x0, dtype=np.float64), lo, hi)
    lam = 1e-3
    eye3 = np.eye(3, dtype=np.float64)
    for _ in range(max_iter):
        d, ds, f = x[0], x[1], x[2]
        ed = np.exp(-bvals * d)
        eds = np.exp(-bvals * ds)
        pred = s0 * (f * eds + (1.0 - f) * ed)
        r = pred - y
        cost = 0.5 * float(np.dot(r, r))

        j_d = -bvals * s0 * (1.0 - f) * ed
        j_ds = -bvals * s0 * f * eds
        j_f = s0 * (eds - ed)
        J = np.column_stack([j_d, j_ds, j_f])

        grad = J.T @ r
        if float(np.linalg.norm(grad, ord=np.inf)) < tol:
            break
        H = J.T @ J + lam * eye3
        try:
            step = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            break

        x_new = np.clip(x + step, lo, hi)
        # Enforce D_star >= D
        if x_new[1] < x_new[0]:
            x_new[1] = max(x_new[0], lo[1])

        ed_n = np.exp(-bvals * x_new[0])
        eds_n = np.exp(-bvals * x_new[1])
        r_new = s0 * (x_new[2] * eds_n + (1.0 - x_new[2]) * ed_n) - y
        cost_new = 0.5 * float(np.dot(r_new, r_new))
        if cost_new < cost:
            if float(np.linalg.norm(x_new - x)) < tol:
                x = x_new
                break
            x = x_new
            lam = max(lam * 0.3, 1e-12)
        else:
            lam *= 5.0
            if lam > 1e10:
                break
    return x


@dataclass
class IVIMFitResult(Mapping[str, Any]):
    """Container for fitted IVIM parameters and boundary-railing diagnostics.

    Supports attribute access (``result.D``, ``result.D_star``, ``result.f``),
    dictionary-style lookup (``result["D"]``, ``result["D_star"]``, ``result["f"]``),
    and 3-tuple unpacking (``D, D_star, f = fit_ivim(bvals, signal)``).

    Parameters
    ----------
    D : np.ndarray or float
        Estimated slow parenchymal tissue diffusivity :math:`D` in
        :math:`\\text{mm}^2/\\text{s}`. Shape matches the spatial dimensions of
        the input ``signal`` (scalar ``float`` for 1D single-voxel input).
    D_star : np.ndarray or float
        Estimated fast microvascular pseudo-diffusion coefficient :math:`D^*` in
        :math:`\\text{mm}^2/\\text{s}`. Shape matches ``D``.
    f : np.ndarray or float
        Estimated microvascular perfusion fraction :math:`f \\in [0, 1]`
        (dimensionless). Shape matches ``D``.
    S0 : np.ndarray or float
        Baseline :math:`b = 0` signal intensity :math:`S_0`. Shape matches ``D``.
    railed_mask : np.ndarray or bool
        Boolean flag or mask indicating voxels whose :math:`D^*` estimate
        terminated within ``rail_tol`` of the lower or upper :math:`D^*` bound.
    lower_mask : np.ndarray or bool
        Boolean flag or mask indicating lower-bound :math:`D^*` railing.
    upper_mask : np.ndarray or bool
        Boolean flag or mask indicating upper-bound :math:`D^*` railing.
    method : str
        Fitting algorithm used (``"segmented"`` or ``"nlls"``).
    bounds : dict of str to tuple of (float, float)
        Optimization box bounds applied to ``D``, ``D_star``, and ``f``.
    b_thresh : float
        Threshold :math:`b`-value (:math:`\\text{s}/\\text{mm}^2`) separating the
        high-:math:`b` mono-exponential tissue regime from the low-:math:`b`
        perfusion regime.
    affine : np.ndarray or None, optional
        4x4 NIfTI spatial affine matrix when a NIfTI image was supplied.
    """

    D: np.ndarray | float
    D_star: np.ndarray | float
    f: np.ndarray | float
    S0: np.ndarray | float
    railed_mask: np.ndarray | bool
    lower_mask: np.ndarray | bool
    upper_mask: np.ndarray | bool
    method: str = "segmented"
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    b_thresh: float = DEFAULT_B_THRESH
    affine: np.ndarray | None = None

    @property
    def Dstar(self) -> np.ndarray | float:
        """Alias for :attr:`D_star`."""
        return self.D_star

    @property
    def dstar(self) -> np.ndarray | float:
        """Lowercase alias for :attr:`D_star`."""
        return self.D_star

    @property
    def railed(self) -> np.ndarray | bool:
        """Alias for :attr:`railed_mask`."""
        return self.railed_mask

    @property
    def railed_lo(self) -> np.ndarray | bool:
        """Alias for :attr:`lower_mask`."""
        return self.lower_mask

    @property
    def railed_hi(self) -> np.ndarray | bool:
        """Alias for :attr:`upper_mask`."""
        return self.upper_mask

    @property
    def frac_railed(self) -> float:
        """Fraction of finite fitted voxels whose :math:`D^*` estimate railed."""
        arr = np.asarray(self.D_star, dtype=np.float64)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return float("nan")
        r = np.asarray(self.railed_mask, dtype=bool)
        return float(np.mean(r[finite]))

    def __iter__(self) -> Iterator[ Any ]:
        """Yield ``(D, D_star, f)`` for convenient tuple unpacking."""
        yield self.D
        yield self.D_star
        yield self.f

    def __len__(self) -> int:
        return 7

    def keys(self) -> list[str]:  # type: ignore[override]
        return [
            "D",
            "D_star",
            "f",
            "S0",
            "railed_mask",
            "lower_mask",
            "upper_mask",
        ]

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            tuple_items = (self.D, self.D_star, self.f)
            return tuple_items[key]
        aliases = {
            "D": self.D,
            "D_star": self.D_star,
            "Dstar": self.D_star,
            "dstar": self.D_star,
            "f": self.f,
            "S0": self.S0,
            "railed": self.railed_mask,
            "railed_mask": self.railed_mask,
            "lower_mask": self.lower_mask,
            "upper_mask": self.upper_mask,
            "railed_lo": self.lower_mask,
            "railed_hi": self.upper_mask,
        }
        if key in aliases:
            return aliases[key]
        raise KeyError(f"Unknown key {key!r} in IVIMFitResult")

    def to_dict(self) -> dict[str, Any]:
        """Return a standard Python dictionary of fitted parameter maps and masks."""
        return {
            "D": self.D,
            "D_star": self.D_star,
            "f": self.f,
            "S0": self.S0,
            "railed_mask": self.railed_mask,
            "lower_mask": self.lower_mask,
            "upper_mask": self.upper_mask,
            "frac_railed": self.frac_railed,
            "method": self.method,
            "bounds": self.bounds,
            "b_thresh": self.b_thresh,
        }

    def to_nifti(self, param: Literal["D", "D_star", "f", "S0", "railed_mask"] = "D") -> Any:
        """Convert a fitted parameter map into a :class:`nibabel.Nifti1Image`.

        Parameters
        ----------
        param : {"D", "D_star", "f", "S0", "railed_mask"}, default="D"
            Name of the parameter map to wrap as a NIfTI image.

        Returns
        -------
        nibabel.Nifti1Image
            NIfTI image containing the requested parameter map with the stored
            spatial affine (or identity affine if none was provided).
        """
        if nib is None:
            raise ImportError("nibabel is required to export NIfTI images.")
        data = np.asarray(self[param], dtype=np.float32)
        affine = self.affine if self.affine is not None else np.eye(4, dtype=np.float64)
        return nib.Nifti1Image(data, affine)


def fit_segmented_voxel(
    bvals: np.ndarray | Sequence[float],
    signal: np.ndarray | Sequence[float],
    b_thresh: float = DEFAULT_B_THRESH,
    bounds_D: tuple[float, float] = DEFAULT_BOUNDS_D,
    bounds_f: tuple[float, float] = DEFAULT_BOUNDS_F,
    bounds_D_star: tuple[float, float] = DEFAULT_BOUNDS_D_STAR,
    bounds_Dstar: tuple[float, float] | None = None,
) -> tuple[float, float, float]:
    """Fit a single voxel's IVIM decay curve using the two-step segmented algorithm.

    Step 1 fits a mono-exponential tissue decay :math:`S(b) = S_{\\text{int}} \\exp(-b D)`
    over high :math:`b`-values (:math:`b \\ge b_{\\text{thresh}}`) via log-linear
    least squares and estimates the perfusion fraction as
    :math:`f = 1 - S_{\\text{int}} / S(b_{\\min})`.
    Step 2 holds :math:`(\\hat{D}, \\hat{f})` fixed and fits :math:`D^*` over all
    :math:`b`-values via bounded Trust-Region Reflective non-linear least squares
    subject to :math:`\\max(D^*_{\\min}, \\hat{D}) \\le D^* \\le D^*_{\\max}`.

    Parameters
    ----------
    bvals : np.ndarray or Sequence[float]
        1D array of diffusion :math:`b`-values in :math:`\\text{s}/\\text{mm}^2`
        with shape ``(N_b,)``.
    signal : np.ndarray or Sequence[float]
        1D array of measured DWI signal intensities at ``bvals`` with shape
        ``(N_b,)``.
    b_thresh : float, default=200.0
        Threshold :math:`b`-value (:math:`\\text{s}/\\text{mm}^2`) separating the
        slow tissue diffusion regime (:math:`b \\ge b_{\\text{thresh}}`) from the
        low-:math:`b` microvascular perfusion regime.
    bounds_D : tuple of (float, float), default=(1e-4, 5e-3)
        Lower and upper physiological bounds for tissue diffusivity :math:`D`
        in :math:`\\text{mm}^2/\\text{s}`.
    bounds_f : tuple of (float, float), default=(0.0, 0.7)
        Lower and upper bounds for perfusion fraction :math:`f` (dimensionless).
    bounds_D_star : tuple of (float, float), default=(3e-3, 0.15)
        Lower and upper bounds for pseudo-diffusion coefficient :math:`D^*`
        in :math:`\\text{mm}^2/\\text{s}`.
    bounds_Dstar : tuple of (float, float) or None, optional
        Backward-compatible alias for ``bounds_D_star``.

    Returns
    -------
    D_hat : float
        Fitted tissue diffusivity :math:`D` (:math:`\\text{mm}^2/\\text{s}`).
    D_star_hat : float
        Fitted pseudo-diffusion coefficient :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
    f_hat : float
        Fitted perfusion fraction :math:`f` (dimensionless).
    """
    if bounds_Dstar is not None:
        bounds_D_star = bounds_Dstar

    b = np.asarray(bvals, dtype=np.float64)
    sig = np.asarray(signal, dtype=np.float64)

    if b.ndim != 1 or sig.ndim != 1 or b.size != sig.size:
        raise ValueError(
            f"bvals and signal must be 1D arrays of equal length; "
            f"got shapes {b.shape} and {sig.shape}."
        )
    if not np.all(np.isfinite(sig)) or np.all(sig <= 0):
        return float("nan"), float("nan"), float("nan")

    # Normalize by mean signal at the minimum available b-value (typically b=0)
    b_min = float(np.min(b))
    min_mask = np.isclose(b, b_min, atol=1e-6)
    s0 = max(float(np.mean(sig[min_mask])), 1e-12)
    y = np.clip(sig / s0, 1e-9, None)

    # Step 1: High-b mono-exponential fit for D and f
    high_mask = b >= b_thresh
    if int(np.sum(high_mask)) < 2:
        sorted_idx = np.argsort(b)
        high_mask = np.zeros_like(b, dtype=bool)
        high_mask[sorted_idx[-2:]] = True

    b_high = b[high_mask]
    log_y_high = np.log(y[high_mask])

    design = np.vstack([np.ones_like(b_high), -b_high]).T
    coeffs, _, _, _ = np.linalg.lstsq(design, log_y_high, rcond=None)
    log_s_int, d_raw = float(coeffs[0]), float(coeffs[1])

    d_hat = float(np.clip(d_raw, bounds_D[0], bounds_D[1]))
    s_int = float(np.exp(log_s_int))
    # Account for non-zero b_min when normalizing at b_min
    f_raw = 1.0 - s_int * float(np.exp(-b_min * d_hat))
    f_hat = float(np.clip(f_raw, bounds_f[0], bounds_f[1]))

    # Step 2: 1D bounded NLLS fit for D* holding (d_hat, f_hat) fixed
    dstar_lo, dstar_hi = float(bounds_D_star[0]), float(bounds_D_star[1])
    effective_lo = max(dstar_lo, d_hat)
    if effective_lo >= dstar_hi:
        return d_hat, dstar_hi, f_hat

    dstar_init = float(np.clip(0.015, effective_lo + 1e-5, dstar_hi - 1e-5))
    tail = (1.0 - f_hat) * np.exp(-b * d_hat)

    dstar_hat = _bounded_lm_1d(
        bvals=b,
        y=y,
        tail=tail,
        f_hat=f_hat,
        x0=dstar_init,
        lo=effective_lo,
        hi=dstar_hi,
    )
    return d_hat, dstar_hat, f_hat


def fit_nlls_voxel(
    bvals: np.ndarray | Sequence[float],
    signal: np.ndarray | Sequence[float],
    bounds_D: tuple[float, float] = DEFAULT_BOUNDS_D,
    bounds_D_star: tuple[float, float] = DEFAULT_BOUNDS_D_STAR,
    bounds_f: tuple[float, float] = DEFAULT_BOUNDS_F,
    b_thresh: float = DEFAULT_B_THRESH,
) -> tuple[float, float, float]:
    """Fit a single voxel's IVIM decay curve via simultaneous bounded NLLS.

    Warm-starts from the two-step segmented estimate and jointly optimizes
    :math:`(D, D^*, f)` using bounded Trust-Region Reflective non-linear
    least squares.

    Parameters
    ----------
    bvals : np.ndarray or Sequence[float]
        1D array of diffusion :math:`b`-values in :math:`\\text{s}/\\text{mm}^2`
        with shape ``(N_b,)``.
    signal : np.ndarray or Sequence[float]
        1D array of measured DWI signal intensities at ``bvals`` with shape
        ``(N_b,)``.
    bounds_D : tuple of (float, float), default=(1e-4, 5e-3)
        Lower and upper bounds for tissue diffusivity :math:`D` (:math:`\\text{mm}^2/\\text{s}`).
    bounds_D_star : tuple of (float, float), default=(3e-3, 0.15)
        Lower and upper bounds for pseudo-diffusion :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
    bounds_f : tuple of (float, float), default=(0.0, 0.7)
        Lower and upper bounds for perfusion fraction :math:`f` (dimensionless).
    b_thresh : float, default=200.0
        Threshold :math:`b`-value (:math:`\\text{s}/\\text{mm}^2`) used for the
        segmented warm start.

    Returns
    -------
    D_hat : float
        Fitted tissue diffusivity :math:`D` (:math:`\\text{mm}^2/\\text{s}`).
    D_star_hat : float
        Fitted pseudo-diffusion coefficient :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
    f_hat : float
        Fitted perfusion fraction :math:`f` (dimensionless).
    """
    b = np.asarray(bvals, dtype=np.float64)
    sig = np.asarray(signal, dtype=np.float64)
    if b.ndim != 1 or sig.ndim != 1 or b.size != sig.size:
        raise ValueError(
            f"bvals and signal must be 1D arrays of equal length; "
            f"got shapes {b.shape} and {sig.shape}."
        )
    if not np.all(np.isfinite(sig)) or np.all(sig <= 0):
        return float("nan"), float("nan"), float("nan")

    b_min = float(np.min(b))
    min_mask = np.isclose(b, b_min, atol=1e-6)
    s0 = max(float(np.mean(sig[min_mask])), 1e-12)
    y = sig / s0

    d0, ds0, f0 = fit_segmented_voxel(
        bvals=b,
        signal=y,
        b_thresh=b_thresh,
        bounds_D=bounds_D,
        bounds_f=bounds_f,
        bounds_D_star=bounds_D_star,
    )
    if not (np.isfinite(d0) and np.isfinite(ds0) and np.isfinite(f0)):
        d0, ds0, f0 = 1.2e-3, 0.02, 0.15

    lo = np.array([bounds_D[0], bounds_D_star[0], bounds_f[0]], dtype=np.float64)
    hi = np.array([bounds_D[1], bounds_D_star[1], bounds_f[1]], dtype=np.float64)
    x0 = np.clip([d0, ds0, f0], lo + 1e-7, hi - 1e-7)

    est = _bounded_lm_3d(bvals=b, y=y, x0=x0, lo=lo, hi=hi, s0=1.0)
    return float(est[0]), float(est[1]), float(est[2])


def classify_railing(
    D_star: np.ndarray | float,
    bounds_D_star: tuple[float, float] = DEFAULT_BOUNDS_D_STAR,
    rel_tol: float = DEFAULT_RAIL_TOL,
    bounds_Dstar: tuple[float, float] | None = None,
) -> dict[str, np.ndarray | bool]:
    """Identify voxels whose :math:`D^*` estimate railed to a parameter bound.

    A voxel is classified as railed if its finite :math:`D^*` estimate lies
    within ``rel_tol * (D_star_max - D_star_min)`` of the lower or upper
    optimization bound.

    Parameters
    ----------
    D_star : np.ndarray or float
        Estimated pseudo-diffusion coefficient :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
    bounds_D_star : tuple of (float, float), default=(3e-3, 0.15)
        Lower and upper optimization bounds ``(D_star_min, D_star_max)`` in
        :math:`\\text{mm}^2/\\text{s}`.
    rel_tol : float, default=1e-2
        Relative tolerance as a fraction of the bound span ``D_star_max - D_star_min``.
    bounds_Dstar : tuple of (float, float) or None, optional
        Backward-compatible alias for ``bounds_D_star``.

    Returns
    -------
    dict of str to (np.ndarray or bool)
        Dictionary with keys:
        - ``"railed_mask"``: boolean mask/flag of any boundary railing,
        - ``"lower_mask"``: boolean mask/flag of lower-bound railing,
        - ``"upper_mask"``: boolean mask/flag of upper-bound railing.
    """
    if bounds_Dstar is not None:
        bounds_D_star = bounds_Dstar

    scalar_input = np.ndim(D_star) == 0
    ds = np.asarray(D_star, dtype=np.float64)
    lo, hi = float(bounds_D_star[0]), float(bounds_D_star[1])
    eps = float(rel_tol) * (hi - lo)

    finite = np.isfinite(ds)
    lower_mask = finite & (np.abs(ds - lo) <= eps)
    upper_mask = finite & (np.abs(ds - hi) <= eps)
    railed_mask = lower_mask | upper_mask

    if scalar_input:
        return {
            "railed_mask": bool(railed_mask.item()),
            "lower_mask": bool(lower_mask.item()),
            "upper_mask": bool(upper_mask.item()),
        }
    return {
        "railed_mask": railed_mask,
        "lower_mask": lower_mask,
        "upper_mask": upper_mask,
    }


def _load_image_or_array(
    obj: Any,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract a NumPy float64 array and optional 4x4 affine from array/NIfTI/path."""
    if hasattr(obj, "get_fdata"):
        data = np.asarray(obj.get_fdata(), dtype=np.float64)
        affine = getattr(obj, "affine", None)
        return data, (np.asarray(affine, dtype=np.float64) if affine is not None else None)

    if isinstance(obj, (str, Path)):
        path = Path(obj)
        if not path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {path}")
        if nib is None:
            raise ImportError(f"nibabel is required to load NIfTI file {path}")
        img = nib.load(str(path))
        return np.asarray(img.get_fdata(), dtype=np.float64), np.asarray(img.affine, dtype=np.float64)

    return np.asarray(obj, dtype=np.float64), None


def fit_ivim(
    bvals: np.ndarray | Sequence[float] | Any,
    signal: np.ndarray | Sequence[float] | str | Path | Any | None = None,
    *,
    method: Literal["segmented", "nlls"] = "segmented",
    b_thresh: float = DEFAULT_B_THRESH,
    bounds_D: tuple[float, float] = DEFAULT_BOUNDS_D,
    bounds_D_star: tuple[float, float] = DEFAULT_BOUNDS_D_STAR,
    bounds_f: tuple[float, float] = DEFAULT_BOUNDS_F,
    bounds_Dstar: tuple[float, float] | None = None,
    rail_tol: float = DEFAULT_RAIL_TOL,
    mask: np.ndarray | str | Path | Any | None = None,
    nifti_img: str | Path | Any | None = None,
) -> IVIMFitResult:
    """Fit the bi-exponential IVIM-DWI model across 1D, 2D, 3D, or NIfTI signals.

    Supports both two-step segmented fitting (``method="segmented"``) and
    simultaneous bounded Trust-Region Reflective non-linear least squares
    (``method="nlls"``), and automatically computes boundary-railing masks for
    :math:`D^*`. Completely decoupled from local filesystem paths.

    Parameters
    ----------
    bvals : np.ndarray or Sequence[float] or SyntheticPhantom
        1D array of diffusion :math:`b`-values in :math:`\\text{s}/\\text{mm}^2`
        with shape ``(N_b,)``. If a :class:`~sextant_core.phantom.SyntheticPhantom`
        (or mapping containing ``"bvals"`` and ``"signal"``) is passed as the
        first argument and ``signal`` is ``None``, both ``bvals`` and ``signal``
        are extracted from it automatically.
    signal : np.ndarray, nibabel.Nifti1Image, str, or Path, optional
        Measured DWI signal decay array with shape:
        - ``(N_b,)`` for a single voxel,
        - ``(N_voxels, N_b)`` for a 1D batch of voxels,
        - ``(H, W, N_b)`` for a 2D spatial slice, or
        - ``(H, W, Z, N_b)`` for a 3D volume or 4D NIfTI image.
        May also be passed as a :class:`nibabel.Nifti1Image` instance or a
        filesystem path to a ``.nii`` / ``.nii.gz`` file, or via ``nifti_img``.
    method : {"segmented", "nlls"}, default="segmented"
        Fitting algorithm:
        - ``"segmented"``: OSIPI standard two-step segmented fit (mono-exponential
          log-linear fit on :math:`b \\ge b_{\\text{thresh}}` for :math:`D` and :math:`f`,
          followed by 1D bounded NLLS over all :math:`b`-values for :math:`D^*`).
        - ``"nlls"``: simultaneous 3-parameter bounded NLLS warm-started from
          the segmented estimate.
    b_thresh : float, default=200.0
        Threshold :math:`b`-value (:math:`\\text{s}/\\text{mm}^2`) separating the
        slow tissue diffusion regime (:math:`b \\ge b_{\\text{thresh}}`) from the
        low-:math:`b` microvascular perfusion regime.
    bounds_D : tuple of (float, float), default=(1e-4, 5e-3)
        Lower and upper bounds for tissue diffusivity :math:`D` (:math:`\\text{mm}^2/\\text{s}`).
    bounds_D_star : tuple of (float, float), default=(3e-3, 0.15)
        Lower and upper bounds for pseudo-diffusion :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
    bounds_f : tuple of (float, float), default=(0.0, 0.7)
        Lower and upper bounds for perfusion fraction :math:`f` (dimensionless).
    bounds_Dstar : tuple of (float, float) or None, optional
        Backward-compatible alias for ``bounds_D_star``.
    rail_tol : float, default=1e-2
        Relative tolerance (as a fraction of ``bounds_D_star[1] - bounds_D_star[0]``)
        within which a :math:`D^*` estimate is flagged as boundary-railed.
    mask : np.ndarray, nibabel.Nifti1Image, str, or Path, optional
        Optional boolean or binary ROI mask matching the spatial dimensions
        ``signal.shape[:-1]``. Voxels where ``mask == 0`` are skipped and set to
        ``np.nan`` in the returned parameter maps.
    nifti_img : nibabel.Nifti1Image, str, or Path, optional
        Optional NIfTI image or path supplying the 4D signal volume (if ``signal``
        is ``None``) and/or the spatial affine matrix.

    Returns
    -------
    IVIMFitResult
        Result container holding:
        - ``D``: tissue diffusivity map (:math:`\\text{mm}^2/\\text{s}`),
        - ``D_star``: pseudo-diffusion map (:math:`\\text{mm}^2/\\text{s}`),
        - ``f``: perfusion fraction map (dimensionless in :math:`[0, 1]`),
        - ``S0``: baseline :math:`b = 0` signal intensity map,
        - ``railed_mask``: boolean boundary-railing mask for :math:`D^*`,
        - ``lower_mask``: boolean lower-bound railing mask for :math:`D^*`,
        - ``upper_mask``: boolean upper-bound railing mask for :math:`D^*`.
        Supports attribute access (``res.D``), dictionary lookup (``res["D"]``),
        and tuple unpacking (``D, D_star, f = fit_ivim(bvals, signal)``).

    Raises
    ------
    ValueError
        If the last dimension of ``signal`` does not match ``len(bvals)``, or if
        an unsupported ``method`` is specified.

    Examples
    --------
    >>> import numpy as np
    >>> from sextant_core import fit_ivim, generate_synthetic_phantom
    >>> phantom = generate_synthetic_phantom(D=0.001, D_star=0.02, f=0.15, snr=None)
    >>> res = fit_ivim(phantom.bvals, phantom.signal)
    >>> round(res.D, 4), round(res.D_star, 3), round(res.f, 2)
    (0.001, 0.02, 0.15)
    """
    if bounds_Dstar is not None:
        bounds_D_star = bounds_Dstar

    # Support passing a SyntheticPhantom or dict as the first argument
    if signal is None and nifti_img is None:
        if hasattr(bvals, "bvals") and hasattr(bvals, "signal"):
            signal = bvals.signal
            bvals = bvals.bvals
        elif isinstance(bvals, Mapping) and "bvals" in bvals and "signal" in bvals:
            signal = bvals["signal"]
            bvals = bvals["bvals"]
        else:
            raise ValueError("`signal` (or `nifti_img`) must be provided to fit_ivim().")

    if hasattr(signal, "signal") and hasattr(signal, "bvals"):
        signal = signal.signal

    if signal is None and nifti_img is not None:
        signal = nifti_img

    b = np.asarray(bvals, dtype=np.float64)
    if b.ndim != 1 or b.size < 3:
        raise ValueError(f"`bvals` must be a 1D array with at least 3 values; got shape {b.shape}.")

    sig_arr, affine = _load_image_or_array(signal)
    if affine is None and nifti_img is not None:
        _, affine = _load_image_or_array(nifti_img)

    if sig_arr.ndim == 0 or sig_arr.shape[-1] != b.size:
        raise ValueError(
            f"Last axis of `signal` (shape {sig_arr.shape}) must match "
            f"len(bvals) ({b.size})."
        )

    if method not in ("segmented", "nlls"):
        raise ValueError(f"Unknown fitting method {method!r}; expected 'segmented' or 'nlls'.")

    b_min = float(np.min(b))
    min_b_mask = np.isclose(b, b_min, atol=1e-6)

    bounds_dict = {
        "D": (float(bounds_D[0]), float(bounds_D[1])),
        "D_star": (float(bounds_D_star[0]), float(bounds_D_star[1])),
        "f": (float(bounds_f[0]), float(bounds_f[1])),
    }

    # 1D single-voxel input
    if sig_arr.ndim == 1:
        s0_val = float(np.mean(sig_arr[min_b_mask]))
        if method == "segmented":
            d_val, ds_val, f_val = fit_segmented_voxel(
                bvals=b,
                signal=sig_arr,
                b_thresh=b_thresh,
                bounds_D=bounds_D,
                bounds_f=bounds_f,
                bounds_D_star=bounds_D_star,
            )
        else:
            d_val, ds_val, f_val = fit_nlls_voxel(
                bvals=b,
                signal=sig_arr,
                bounds_D=bounds_D,
                bounds_D_star=bounds_D_star,
                bounds_f=bounds_f,
                b_thresh=b_thresh,
            )
        rail_info = classify_railing(ds_val, bounds_D_star=bounds_D_star, rel_tol=rail_tol)
        return IVIMFitResult(
            D=float(d_val),
            D_star=float(ds_val),
            f=float(f_val),
            S0=s0_val,
            railed_mask=bool(rail_info["railed_mask"]),
            lower_mask=bool(rail_info["lower_mask"]),
            upper_mask=bool(rail_info["upper_mask"]),
            method=method,
            bounds=bounds_dict,
            b_thresh=float(b_thresh),
            affine=affine,
        )

    # Multi-voxel (1D batch, 2D slice, or 3D volume)
    spatial_shape = sig_arr.shape[:-1]
    n_voxels = int(np.prod(spatial_shape))
    flat_sig = sig_arr.reshape(n_voxels, b.size)

    if mask is not None:
        mask_arr, _ = _load_image_or_array(mask)
        if mask_arr.shape != spatial_shape:
            raise ValueError(
                f"`mask` shape {mask_arr.shape} must match spatial shape {spatial_shape}."
            )
        flat_mask = (mask_arr.reshape(n_voxels) > 0) & np.all(np.isfinite(flat_sig), axis=1)
    else:
        flat_mask = np.all(np.isfinite(flat_sig), axis=1) & (np.max(flat_sig, axis=1) > 0)

    d_flat = np.full(n_voxels, np.nan, dtype=np.float64)
    ds_flat = np.full(n_voxels, np.nan, dtype=np.float64)
    f_flat = np.full(n_voxels, np.nan, dtype=np.float64)
    s0_flat = np.full(n_voxels, np.nan, dtype=np.float64)

    active_indices = np.flatnonzero(flat_mask)
    for idx in active_indices:
        vox_sig = flat_sig[idx]
        s0_flat[idx] = float(np.mean(vox_sig[min_b_mask]))
        if method == "segmented":
            d_flat[idx], ds_flat[idx], f_flat[idx] = fit_segmented_voxel(
                bvals=b,
                signal=vox_sig,
                b_thresh=b_thresh,
                bounds_D=bounds_D,
                bounds_f=bounds_f,
                bounds_D_star=bounds_D_star,
            )
        else:
            d_flat[idx], ds_flat[idx], f_flat[idx] = fit_nlls_voxel(
                bvals=b,
                signal=vox_sig,
                bounds_D=bounds_D,
                bounds_D_star=bounds_D_star,
                bounds_f=bounds_f,
                b_thresh=b_thresh,
            )

    d_map = d_flat.reshape(spatial_shape)
    ds_map = ds_flat.reshape(spatial_shape)
    f_map = f_flat.reshape(spatial_shape)
    s0_map = s0_flat.reshape(spatial_shape)

    rail_info = classify_railing(ds_map, bounds_D_star=bounds_D_star, rel_tol=rail_tol)
    return IVIMFitResult(
        D=d_map,
        D_star=ds_map,
        f=f_map,
        S0=s0_map,
        railed_mask=np.asarray(rail_info["railed_mask"], dtype=bool),
        lower_mask=np.asarray(rail_info["lower_mask"], dtype=bool),
        upper_mask=np.asarray(rail_info["upper_mask"], dtype=bool),
        method=method,
        bounds=bounds_dict,
        b_thresh=float(b_thresh),
        affine=affine,
    )


def compute_jacobian(
    bvals: np.ndarray | Sequence[float],
    D: float,
    D_star: float,
    f: float,
    S0: float = 1.0,
) -> np.ndarray:
    """Compute the analytical Jacobian matrix :math:`J \\in \\mathbb{R}^{N_b \\times 3}` of :math:`S(b)` with respect to :math:`(D, D^*, f)`.

    The partial derivatives are:

    .. math::

        \\frac{\\partial S(b)}{\\partial D} &= -b S_0 (1 - f) \\exp(-b D), \\\\
        \\frac{\\partial S(b)}{\\partial D^*} &= -b S_0 f \\exp(-b D^*), \\\\
        \\frac{\\partial S(b)}{\\partial f} &= S_0 \\left[ \\exp(-b D^*) - \\exp(-b D) \\right].

    Parameters
    ----------
    bvals : np.ndarray or Sequence[float]
        1D array of diffusion :math:`b`-values (:math:`\\text{s}/\\text{mm}^2`).
    D : float
        Slow tissue diffusivity :math:`D` (:math:`\\text{mm}^2/\\text{s}`).
    D_star : float
        Fast pseudo-diffusion coefficient :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
    f : float
        Perfusion fraction :math:`f \\in [0, 1]`.
    S0 : float, default=1.0
        Baseline :math:`b = 0` signal intensity :math:`S_0`.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape ``(N_b, 3)`` with columns ``[dS/dD, dS/dD_star, dS/df]``.
    """
    b = np.asarray(bvals, dtype=np.float64)
    ed = np.exp(-b * float(D))
    eds = np.exp(-b * float(D_star))
    s0 = float(S0)
    f_val = float(f)
    j_d = -b * s0 * (1.0 - f_val) * ed
    j_ds = -b * s0 * f_val * eds
    j_f = s0 * (eds - ed)
    return np.column_stack([j_d, j_ds, j_f])


def compute_crlb(
    bvals: np.ndarray | Sequence[float],
    D: float,
    D_star: float,
    f: float,
    snr: float = 50.0,
    S0: float = 1.0,
    reg: float = 1e-12,
) -> dict[str, np.ndarray | float]:
    """Compute the Gaussian Fisher Information Matrix and Cramér–Rao Lower Bound (CRLB).

    At noise standard deviation :math:`\\sigma = S_0 / \\text{SNR}`, the Fisher
    Information Matrix is :math:`\\mathcal{I}(\\theta) = \\sigma^{-2} J^\\top J` and
    the Cramér–Rao lower bound covariance is :math:`\\Sigma_{\\text{CRLB}} = (\\mathcal{I} + \\lambda I)^{-1}`.

    Parameters
    ----------
    bvals : np.ndarray or Sequence[float]
        1D array of diffusion :math:`b`-values (:math:`\\text{s}/\\text{mm}^2`).
    D : float
        Slow tissue diffusivity :math:`D` (:math:`\\text{mm}^2/\\text{s}`).
    D_star : float
        Fast pseudo-diffusion coefficient :math:`D^*` (:math:`\\text{mm}^2/\\text{s}`).
    f : float
        Perfusion fraction :math:`f \\in [0, 1]`.
    snr : float, default=50.0
        Signal-to-noise ratio :math:`S_0 / \\sigma > 0`.
    S0 : float, default=1.0
        Baseline :math:`b = 0` signal intensity :math:`S_0`.
    reg : float, default=1e-12
        Tikhonov diagonal regularization for near-singular Fisher matrices.

    Returns
    -------
    dict
        Dictionary containing:
        - ``"std"``: 1D array ``[sigma_D, sigma_D_star, sigma_f]`` of CRLB standard deviations,
        - ``"cov"``: 3x3 CRLB covariance matrix,
        - ``"fim"``: 3x3 Fisher Information Matrix,
        - ``"cv_D_star"``: CRLB coefficient of variation :math:`\\sigma_{D^*} / D^*`.
    """
    if float(snr) <= 0 or not np.isfinite(snr):
        raise ValueError(f"`snr` must be a positive finite number; got {snr!r}.")
    sigma = float(S0) / float(snr)
    jac = compute_jacobian(bvals, D=D, D_star=D_star, f=f, S0=S0)
    fim = (jac.T @ jac) / (sigma**2)
    cov = np.linalg.inv(fim + float(reg) * np.eye(3, dtype=np.float64))
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return {
        "std": std,
        "cov": cov,
        "fim": fim,
        "cv_D_star": float(std[1] / max(float(D_star), 1e-12)),
    }
