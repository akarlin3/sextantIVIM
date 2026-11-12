"""Pure-NumPy / nibabel NIfTI-1 I/O and multi-core volumetric IVIM fitting.

Provides:
- Standards-compliant 348-byte NIfTI-1 (``.nii`` and ``.nii.gz``) reader and writer
  (:func:`read_nifti`, :func:`write_nifti`, :class:`NiftiVolume`) implemented with
  pure NumPy + ``struct`` + ``gzip`` and seamless ``nibabel`` acceleration when
  installed.
- ``.bval`` and ``.te`` acquisition schedule file helpers (:func:`read_bvals`,
  :func:`write_bvals`, :func:`read_te_values`, :func:`write_te_values`).
- Chunked multi-process volumetric 1D and 2D ``(b, TE)`` IVIM fitting
  (:func:`fit_volume`, :func:`fit_volume_files`, :class:`VolumeFitResult`) using
  :class:`concurrent.futures.ProcessPoolExecutor` with exact bit-for-bit parity
  across worker counts.

License: AGPL-3.0-or-later
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import struct
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np

from .fitting import (
    DEFAULT_B_THRESH,
    DEFAULT_RAIL_TOL,
    classify_railing,
    fit_nlls_voxel,
    fit_segmented_voxel,
)
from .t2_ivim import (
    PARAM_NAMES_5,
    _normalize_bounds,
    _resolve_schedule,
    fit_t2_ivim_voxel,
)

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - optional runtime dependency
    nib = None


# Standard 348-byte NIfTI-1 binary header format (without endian prefix)
_NIFTI1_HDR_FMT = "i10s18sih1sB8h3f4h8f3fh2B4f2i80s24s2h3f3f4f4f4f16s4s"
_NIFTI1_HDR_SIZE = 348
_NIFTI1_MIN_VOX_OFFSET = 352

_DT_CODE_TO_DTYPE: dict[int, tuple[str, int]] = {
    2: ("u1", 8),      # DT_UINT8
    4: ("i2", 16),     # DT_INT16
    8: ("i4", 32),     # DT_INT32
    16: ("f4", 32),    # DT_FLOAT32
    64: ("f8", 64),    # DT_FLOAT64
    256: ("i1", 8),    # DT_INT8
    512: ("u2", 16),   # DT_UINT16
    768: ("u4", 32),   # DT_UINT32
    1024: ("i8", 64),  # DT_INT64
    1280: ("u8", 64),  # DT_UINT64
}

_DTYPE_TO_DT_CODE: dict[np.dtype[Any], tuple[int, int]] = {
    np.dtype("uint8"): (2, 8),
    np.dtype("int16"): (4, 16),
    np.dtype("int32"): (8, 32),
    np.dtype("float32"): (16, 32),
    np.dtype("float64"): (64, 64),
    np.dtype("int8"): (256, 8),
    np.dtype("uint16"): (512, 16),
}


@dataclass
class NiftiVolume:
    """In-memory NIfTI-1 volume container duck-type compatible with ``nibabel.Nifti1Image``.

    Parameters
    ----------
    data : np.ndarray
        N-dimensional voxel array in standard NIfTI spatial order ``(X, Y, Z, ...)``.
    affine : np.ndarray
        ``(4, 4)`` homogeneous voxel-to-world coordinate affine matrix.
    header : dict of str to Any
        Parsed NIfTI-1 header metadata.
    """

    data: np.ndarray
    affine: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    header: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """Spatial/temporal array shape."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype[Any]:
        """Underlying NumPy dtype."""
        return self.data.dtype

    def get_fdata(self, dtype: Any = np.float64) -> np.ndarray:
        """Return floating-point voxel array (matching ``nibabel.Nifti1Image.get_fdata``)."""
        return np.asarray(self.data, dtype=dtype)

    def to_filename(self, path: str | Path, *, use_nibabel: bool | None = None) -> Path:
        """Write this volume to a ``.nii`` or ``.nii.gz`` path."""
        return write_nifti(self, path, use_nibabel=use_nibabel)


def _quat_to_affine(
    qb: float,
    qc: float,
    qd: float,
    qx: float,
    qy: float,
    qz: float,
    pixdim: Sequence[float],
) -> np.ndarray:
    """Reconstruct a 4x4 NIfTI-1 ``qform`` affine matrix from quaternion parameters."""
    b, c, d = float(qb), float(qc), float(qd)
    a2 = 1.0 - (b * b + c * c + d * d)
    if a2 < 1e-7:
        norm = float(np.sqrt(max(b * b + c * c + d * d, 1e-30)))
        b, c, d = b / norm, c / norm, d / norm
        a = 0.0
    else:
        a = float(np.sqrt(a2))

    R = np.array(
        [
            [a * a + b * b - c * c - d * d, 2.0 * (b * c - a * d), 2.0 * (b * d + a * c)],
            [2.0 * (b * c + a * d), a * a + c * c - b * b - d * d, 2.0 * (c * d - a * b)],
            [2.0 * (b * d - a * c), 2.0 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ],
        dtype=np.float64,
    )
    qfac = -1.0 if float(pixdim[0]) < 0.0 else 1.0
    dx = float(pixdim[1]) if float(pixdim[1]) > 0.0 else 1.0
    dy = float(pixdim[2]) if float(pixdim[2]) > 0.0 else 1.0
    dz = (float(pixdim[3]) if float(pixdim[3]) > 0.0 else 1.0) * qfac

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = R * np.array([dx, dy, dz], dtype=np.float64)[None, :]
    affine[:3, 3] = [float(qx), float(qy), float(qz)]
    return affine


def _affine_to_qform(
    affine: np.ndarray,
) -> tuple[float, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Decompose a 4x4 affine into ``(qfac, (dx, dy, dz), (qb, qc, qd), (qx, qy, qz))``."""
    M = np.asarray(affine[:3, :3], dtype=np.float64)
    scales = np.linalg.norm(M, axis=0)
    scales_safe = np.where(scales > 1e-12, scales, 1.0)
    Q = M / scales_safe[None, :]

    if float(np.linalg.det(Q)) < 0.0:
        qfac = -1.0
        Q = Q.copy()
        Q[:, 2] = -Q[:, 2]
    else:
        qfac = 1.0

    U, _, Vt = np.linalg.svd(Q)
    R = U @ Vt
    if float(np.linalg.det(R)) < 0.0:
        U = U.copy()
        U[:, -1] = -U[:, -1]
        R = U @ Vt

    # Shepherd's method for rotation matrix -> unit quaternion (a >= 0, b, c, d)
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = 2.0 * float(np.sqrt(trace + 1.0))
        a = 0.25 * s
        qb = float(R[2, 1] - R[1, 2]) / s
        qc = float(R[0, 2] - R[2, 0]) / s
        qd = float(R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * float(np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-30)))
        a = float(R[2, 1] - R[1, 2]) / s
        qb = 0.25 * s
        qc = float(R[0, 1] + R[1, 0]) / s
        qd = float(R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * float(np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-30)))
        a = float(R[0, 2] - R[2, 0]) / s
        qb = float(R[0, 1] + R[1, 0]) / s
        qc = 0.25 * s
        qd = float(R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * float(np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-30)))
        a = float(R[1, 0] - R[0, 1]) / s
        qb = float(R[0, 2] + R[2, 0]) / s
        qc = float(R[1, 2] + R[2, 1]) / s
        qd = 0.25 * s

    if a < 0.0:
        qb, qc, qd = -qb, -qc, -qd

    dx, dy, dz = float(scales_safe[0]), float(scales_safe[1]), float(scales_safe[2])
    qx, qy, qz = float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3])
    return qfac, (dx, dy, dz), (qb, qc, qd), (qx, qy, qz)


def _should_use_nibabel(prefer_nibabel: bool, use_nibabel: bool | None) -> bool:
    """Resolve whether to use ``nibabel`` or the pure-NumPy NIfTI-1 implementation."""
    if use_nibabel is True:
        if nib is None:
            raise ImportError("`use_nibabel=True` requested, but `nibabel` is not installed.")
        return True
    if use_nibabel is False:
        return False
    return bool(prefer_nibabel and (nib is not None))


def _is_gzip_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii.gz") or name.endswith(".gz")


def _read_nifti_pure_numpy(path: Path, dtype: Any | None = None) -> NiftiVolume:
    """Read a ``.nii`` or ``.nii.gz`` NIfTI-1 file using pure Python ``struct`` + ``gzip`` + NumPy."""
    with open(path, "rb") as raw_fh:
        magic2 = raw_fh.read(2)
    opener = gzip.open if (magic2 == b"\x1f\x8b" or _is_gzip_path(path)) else open

    with opener(path, "rb") as fh:
        hdr_bytes = fh.read(_NIFTI1_HDR_SIZE)
        if len(hdr_bytes) < _NIFTI1_HDR_SIZE:
            raise ValueError(
                f"File {path} is too short ({len(hdr_bytes)} bytes) for a 348-byte NIfTI-1 header."
            )

        sizeof_le = struct.unpack("<i", hdr_bytes[:4])[0]
        sizeof_be = struct.unpack(">i", hdr_bytes[:4])[0]
        if sizeof_le == _NIFTI1_HDR_SIZE:
            endian = "<"
        elif sizeof_be == _NIFTI1_HDR_SIZE:
            endian = ">"
        else:
            raise ValueError(
                f"Invalid NIfTI-1 header in {path}: sizeof_hdr={sizeof_le} (expected 348)."
            )

        unpacked = struct.unpack(endian + _NIFTI1_HDR_FMT, hdr_bytes)
        dim = unpacked[7:15]
        ndim = int(dim[0])
        if not (1 <= ndim <= 7):
            raise ValueError(f"Invalid NIfTI-1 ndim={ndim} in {path}.")
        shape = tuple(max(int(d), 1) for d in dim[1 : ndim + 1])

        datatype_code = int(unpacked[19])
        bitpix = int(unpacked[20])
        pixdim = tuple(float(x) for x in unpacked[22:30])
        vox_offset = float(unpacked[30])
        scl_slope = float(unpacked[31])
        scl_inter = float(unpacked[32])
        descrip = unpacked[42].split(b"\x00", 1)[0].decode("latin1", errors="replace")
        qform_code = int(unpacked[44])
        sform_code = int(unpacked[45])
        qb, qc, qd = float(unpacked[46]), float(unpacked[47]), float(unpacked[48])
        qx, qy, qz = float(unpacked[49]), float(unpacked[50]), float(unpacked[51])
        srow_x = unpacked[52:56]
        srow_y = unpacked[56:60]
        srow_z = unpacked[60:64]
        magic = unpacked[65]

        if magic not in (b"n+1\x00", b"ni1\x00"):
            raise ValueError(f"Unsupported NIfTI-1 magic bytes {magic!r} in {path}.")

        if datatype_code not in _DT_CODE_TO_DTYPE:
            raise ValueError(f"Unsupported NIfTI-1 datatype code {datatype_code} in {path}.")
        dt_str, expected_bitpix = _DT_CODE_TO_DTYPE[datatype_code]
        disk_dtype = np.dtype(endian + dt_str)

        offset_bytes = max(int(round(vox_offset)), _NIFTI1_HDR_SIZE)
        skip = offset_bytes - _NIFTI1_HDR_SIZE
        if skip > 0:
            fh.read(skip)

        n_elems = int(np.prod(shape))
        n_bytes = n_elems * disk_dtype.itemsize
        payload = fh.read(n_bytes)
        if len(payload) < n_bytes:
            raise ValueError(
                f"Truncated NIfTI-1 voxel payload in {path}: expected {n_bytes} bytes, got {len(payload)}."
            )

    raw_1d = np.frombuffer(payload, dtype=disk_dtype, count=n_elems)
    # NIfTI-1 stores dim[1] fastest on disk (Fortran column-major order)
    arr_native = np.reshape(raw_1d, shape, order="F").astype(
        np.dtype(dt_str), copy=True
    )

    slope_active = np.isfinite(scl_slope) and scl_slope != 0.0
    inter_active = np.isfinite(scl_inter) and scl_inter != 0.0
    if slope_active and (scl_slope != 1.0 or inter_active):
        out_arr = arr_native.astype(np.float64) * scl_slope + (scl_inter if inter_active else 0.0)
    elif not slope_active and inter_active:
        out_arr = arr_native
    else:
        out_arr = arr_native

    if dtype is not None:
        out_arr = np.asarray(out_arr, dtype=dtype)

    if sform_code > 0:
        affine = np.eye(4, dtype=np.float64)
        affine[0, :] = [float(v) for v in srow_x]
        affine[1, :] = [float(v) for v in srow_y]
        affine[2, :] = [float(v) for v in srow_z]
    elif qform_code > 0:
        affine = _quat_to_affine(qb, qc, qd, qx, qy, qz, pixdim)
    else:
        affine = np.eye(4, dtype=np.float64)
        for ax in range(min(3, ndim)):
            step = pixdim[ax + 1] if pixdim[ax + 1] > 0.0 else 1.0
            affine[ax, ax] = step

    header_info: dict[str, Any] = {
        "sizeof_hdr": _NIFTI1_HDR_SIZE,
        "ndim": ndim,
        "dim": list(dim),
        "datatype": datatype_code,
        "bitpix": bitpix or expected_bitpix,
        "pixdim": list(pixdim),
        "vox_offset": offset_bytes,
        "scl_slope": scl_slope,
        "scl_inter": scl_inter,
        "qform_code": qform_code,
        "sform_code": sform_code,
        "descrip": descrip,
        "backend": "numpy",
    }
    return NiftiVolume(data=out_arr, affine=affine, header=header_info)


def _write_nifti_pure_numpy(
    arr: np.ndarray,
    path: Path,
    affine: np.ndarray,
    out_dtype: np.dtype[Any],
    scl_slope: float,
    scl_inter: float,
    descrip: str,
) -> Path:
    """Write a ``.nii`` or ``.nii.gz`` NIfTI-1 file using pure Python ``struct`` + ``gzip`` + NumPy."""
    if arr.ndim < 1 or arr.ndim > 7:
        raise ValueError(f"NIfTI-1 supports 1D to 7D arrays; got {arr.ndim}D shape {arr.shape}.")
    if out_dtype not in _DTYPE_TO_DT_CODE:
        raise ValueError(
            f"Unsupported NIfTI-1 output dtype {out_dtype}; supported: "
            f"{[str(k) for k in _DTYPE_TO_DT_CODE]}."
        )
    if not (np.isfinite(scl_slope) and scl_slope != 0.0):
        raise ValueError(f"`scl_slope` must be finite and non-zero; got {scl_slope!r}.")

    dt_code, bitpix = _DTYPE_TO_DT_CODE[out_dtype]

    if scl_slope != 1.0 or scl_inter != 0.0:
        unscaled = (np.asarray(arr, dtype=np.float64) - float(scl_inter)) / float(scl_slope)
        if np.issubdtype(out_dtype, np.integer):
            info = np.iinfo(out_dtype)
            unscaled = np.clip(np.rint(unscaled), info.min, info.max)
        disk_arr = np.asarray(unscaled, dtype=out_dtype)
    else:
        disk_arr = np.asarray(arr, dtype=out_dtype)

    le_dtype = disk_arr.dtype.newbyteorder("<")
    disk_le = np.asfortranarray(disk_arr, dtype=le_dtype)
    payload_bytes = disk_le.tobytes(order="F")

    dim = [int(arr.ndim)] + [int(s) for s in arr.shape] + [1] * (7 - arr.ndim)
    qfac, (dx, dy, dz), (qb, qc, qd), (qx, qy, qz) = _affine_to_qform(affine)
    pixdim = [float(qfac), dx, dy, dz, 1.0, 1.0, 1.0, 1.0]
    for k in range(arr.ndim + 1, 8):
        pixdim[k] = 0.0

    descrip_bytes = descrip.encode("ascii", errors="replace")[:79].ljust(80, b"\x00")
    srow_x = [float(v) for v in affine[0, :]]
    srow_y = [float(v) for v in affine[1, :]]
    srow_z = [float(v) for v in affine[2, :]]

    hdr_bytes = struct.pack(
        "<" + _NIFTI1_HDR_FMT,
        _NIFTI1_HDR_SIZE,            # sizeof_hdr
        b"\x00" * 10,                # data_type
        b"\x00" * 18,                # db_name
        0,                           # extents
        0,                           # session_error
        b"r",                        # regular
        0,                           # dim_info
        *dim,                        # dim[0..7]
        0.0, 0.0, 0.0,               # intent_p1..3
        0,                           # intent_code
        dt_code,                     # datatype
        bitpix,                      # bitpix
        0,                           # slice_start
        *pixdim,                     # pixdim[0..7]
        float(_NIFTI1_MIN_VOX_OFFSET),  # vox_offset (352.0)
        float(scl_slope),            # scl_slope
        float(scl_inter),            # scl_inter
        0,                           # slice_end
        0,                           # slice_code
        2,                           # xyzt_units (NIFTI_UNITS_MM)
        0.0,                         # cal_max
        0.0,                         # cal_min
        0.0,                         # slice_duration
        0.0,                         # toffset
        0,                           # glmax
        0,                           # glmin
        descrip_bytes,               # descrip
        b"\x00" * 24,                # aux_file
        1,                           # qform_code (NIFTI_XFORM_SCANNER_ANAT)
        1,                           # sform_code (NIFTI_XFORM_SCANNER_ANAT)
        qb, qc, qd,                  # quatern_b..d
        qx, qy, qz,                  # qoffset_x..z
        *srow_x,                     # srow_x[0..3]
        *srow_y,                     # srow_y[0..3]
        *srow_z,                     # srow_z[0..3]
        b"\x00" * 16,                # intent_name
        b"n+1\x00",                  # magic
    )
    extender = b"\x00\x00\x00\x00"

    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_gzip_path(path):
        with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as gzf:
            gzf.write(hdr_bytes)
            gzf.write(extender)
            gzf.write(payload_bytes)
    else:
        with open(path, "wb") as fh:
            fh.write(hdr_bytes)
            fh.write(extender)
            fh.write(payload_bytes)
    return path


def read_nifti(
    path: str | Path,
    *,
    prefer_nibabel: bool = True,
    use_nibabel: bool | None = None,
    dtype: Any | None = None,
) -> NiftiVolume:
    """Read a ``.nii`` or ``.nii.gz`` NIfTI-1 image into a :class:`NiftiVolume`.

    Uses ``nibabel`` when available (``prefer_nibabel=True``) and falls back to
    the pure-NumPy 348-byte NIfTI-1 parser when ``nibabel`` is absent or when
    ``use_nibabel=False`` is explicitly passed.

    Parameters
    ----------
    path : str or Path
        Filesystem path to a ``.nii`` or ``.nii.gz`` file.
    prefer_nibabel : bool, default=True
        Prefer ``nibabel`` when installed and ``use_nibabel`` is ``None``.
    use_nibabel : bool or None, optional
        Explicit backend selector: ``True`` forces ``nibabel``, ``False`` forces
        the pure-NumPy NIfTI-1 reader.
    dtype : numpy dtype or None, optional
        Optional target dtype for ``volume.data``. When ``None``, preserves native
        disk dtype for unscaled data (or ``float64`` when ``scl_slope``/``scl_inter``
        scaling is active).

    Returns
    -------
    NiftiVolume
        Loaded volume with ``data``, ``affine`` (4x4 ``float64``), and ``header``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NIfTI file not found: {p}")

    if _should_use_nibabel(prefer_nibabel=prefer_nibabel, use_nibabel=use_nibabel):
        img = nib.load(str(p))
        hdr = img.header
        slope, inter = hdr.get_slope_inter()
        has_scaling = (slope is not None and slope != 1.0) or (inter is not None and inter != 0.0)
        if dtype is not None:
            if np.issubdtype(np.dtype(dtype), np.floating):
                arr = np.asarray(img.get_fdata(dtype=np.dtype(dtype)), dtype=dtype)
            else:
                arr = np.asarray(img.dataobj, dtype=dtype)
        elif has_scaling:
            arr = np.asarray(img.get_fdata(dtype=np.float64), dtype=np.float64)
        else:
            arr = np.asarray(img.dataobj)
        affine = np.asarray(img.affine, dtype=np.float64)
        header_meta = {
            "sizeof_hdr": int(hdr["sizeof_hdr"]),
            "ndim": int(hdr["dim"][0]),
            "dim": [int(x) for x in hdr["dim"]],
            "datatype": int(hdr["datatype"]),
            "bitpix": int(hdr["bitpix"]),
            "pixdim": [float(x) for x in hdr["pixdim"]],
            "vox_offset": int(hdr["vox_offset"]),
            "scl_slope": 1.0 if slope is None else float(slope),
            "scl_inter": 0.0 if inter is None else float(inter),
            "qform_code": int(hdr["qform_code"]),
            "sform_code": int(hdr["sform_code"]),
            "descrip": bytes(hdr["descrip"]).split(b"\x00", 1)[0].decode("latin1", errors="replace"),
            "backend": "nibabel",
        }
        return NiftiVolume(data=arr, affine=affine, header=header_meta)

    return _read_nifti_pure_numpy(p, dtype=dtype)


def _infer_nifti_dtype(arr: np.ndarray, requested_dtype: Any | None) -> np.dtype[Any]:
    """Select a valid NIfTI-1 disk dtype while preserving precision by default."""
    if requested_dtype is not None:
        dt = np.dtype(requested_dtype)
        if dt == np.dtype("bool"):
            return np.dtype("uint8")
        return dt

    src_dt = arr.dtype
    if src_dt == np.dtype("bool"):
        return np.dtype("uint8")
    if src_dt in _DTYPE_TO_DT_CODE:
        return src_dt
    if np.issubdtype(src_dt, np.signedinteger):
        return np.dtype("int32")
    if np.issubdtype(src_dt, np.unsignedinteger):
        return np.dtype("uint16")
    return np.dtype("float64")


def write_nifti(
    data: np.ndarray | NiftiVolume | str | Path | Any,
    path: str | Path | np.ndarray | NiftiVolume | Any,
    affine: np.ndarray | None = None,
    *,
    dtype: Any | None = None,
    scl_slope: float = 1.0,
    scl_inter: float = 0.0,
    descrip: str = "sextant-core v1.1.0",
    prefer_nibabel: bool = True,
    use_nibabel: bool | None = None,
) -> Path:
    """Write an array or :class:`NiftiVolume` to a ``.nii`` or ``.nii.gz`` file.

    Accepts ``(data, path, affine)`` (primary) as well as ``(path, data, affine)``
    for convenience.

    Parameters
    ----------
    data : np.ndarray or NiftiVolume
        1D–7D voxel array (or :class:`NiftiVolume` / ``nibabel.Nifti1Image``).
    path : str or Path
        Destination ``.nii`` or ``.nii.gz`` path.
    affine : np.ndarray, optional
        ``(4, 4)`` spatial affine matrix. Defaults to ``data.affine`` if present,
        otherwise ``np.eye(4)``.
    dtype : numpy dtype, optional
        Explicit output NIfTI-1 dtype (``float32``, ``float64``, ``int16``, ``uint8``, etc.).
    scl_slope : float, default=1.0
        NIfTI-1 linear data scaling slope.
    scl_inter : float, default=0.0
        NIfTI-1 linear data scaling intercept.
    descrip : str, default="sextant-core v1.1.0"
        Up-to-80-byte ASCII header description string.
    prefer_nibabel : bool, default=True
        Prefer ``nibabel`` when installed and ``use_nibabel`` is ``None``.
    use_nibabel : bool or None, optional
        Explicit backend selector: ``True`` forces ``nibabel``, ``False`` forces
        the pure-NumPy NIfTI-1 writer.

    Returns
    -------
    Path
        Written output file path.
    """
    # Support both write_nifti(data, path, ...) and write_nifti(path, data, ...)
    if isinstance(data, (str, Path)) and not isinstance(path, (str, Path)):
        data_obj, out_path = path, Path(data)
    elif isinstance(path, (str, Path)):
        data_obj, out_path = data, Path(path)
    else:
        raise TypeError("`path` must be a `str` or `pathlib.Path`.")

    if isinstance(data_obj, NiftiVolume):
        arr = np.asarray(data_obj.data)
        aff = np.asarray(data_obj.affine if affine is None else affine, dtype=np.float64)
    elif hasattr(data_obj, "get_fdata"):
        arr = np.asarray(data_obj.dataobj if hasattr(data_obj, "dataobj") else data_obj.get_fdata())
        img_aff = getattr(data_obj, "affine", None)
        aff = np.asarray(
            img_aff if (affine is None and img_aff is not None) else (np.eye(4) if affine is None else affine),
            dtype=np.float64,
        )
    else:
        arr = np.asarray(data_obj)
        aff = np.eye(4, dtype=np.float64) if affine is None else np.asarray(affine, dtype=np.float64)

    if aff.shape != (4, 4):
        raise ValueError(f"`affine` must have shape (4, 4); got {aff.shape}.")

    out_dtype = _infer_nifti_dtype(arr, dtype)

    # When explicit non-unit slope/intercept are requested, the pure-NumPy writer
    # guarantees exact header scl_slope / scl_inter encoding across dtypes.
    if _should_use_nibabel(prefer_nibabel=prefer_nibabel, use_nibabel=use_nibabel) and (
        scl_slope == 1.0 and scl_inter == 0.0
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cast_arr = np.asarray(arr, dtype=out_dtype)
        img = nib.Nifti1Image(cast_arr, aff)
        img.header.set_data_dtype(out_dtype)
        img.header.set_sform(aff, code=1)
        img.header.set_qform(aff, code=1)
        img.header["descrip"] = descrip.encode("ascii", errors="replace")[:79]
        nib.save(img, str(out_path))
        return out_path

    return _write_nifti_pure_numpy(
        arr=arr,
        path=out_path,
        affine=aff,
        out_dtype=out_dtype,
        scl_slope=float(scl_slope),
        scl_inter=float(scl_inter),
        descrip=descrip,
    )


def _read_numeric_schedule(path: str | Path, label: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} file not found: {p}")
    tokens: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].replace(",", " ").strip()
        if stripped:
            tokens.extend(stripped.split())
    if not tokens:
        raise ValueError(f"{label} file {p} is empty.")
    vals = np.array([float(tok) for tok in tokens], dtype=np.float64)
    if np.any(~np.isfinite(vals)) or np.any(vals < 0.0):
        raise ValueError(f"{label} values in {p} must be finite and non-negative.")
    return vals


def _write_numeric_schedule(
    values_or_path: np.ndarray | Sequence[float] | str | Path,
    path_or_values: str | Path | np.ndarray | Sequence[float],
) -> Path:
    if isinstance(values_or_path, (str, Path)) and not isinstance(path_or_values, (str, Path)):
        out_path = Path(values_or_path)
        vals = np.asarray(path_or_values, dtype=np.float64).ravel()
    elif isinstance(path_or_values, (str, Path)):
        out_path = Path(path_or_values)
        vals = np.asarray(values_or_path, dtype=np.float64).ravel()
    else:
        raise TypeError("A target file `path` (`str` or `Path`) is required.")

    if vals.size == 0 or np.any(~np.isfinite(vals)):
        raise ValueError("Schedule values must be non-empty and finite.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    formatted = " ".join(f"{float(v):g}" for v in vals) + "\n"
    out_path.write_text(formatted, encoding="utf-8")
    return out_path


def read_bvals(path: str | Path) -> np.ndarray:
    """Read a whitespace- or comma-delimited ``.bval`` file into a 1D ``float64`` array (s/mm^2)."""
    return _read_numeric_schedule(path, "b-value")


def write_bvals(
    bvals: np.ndarray | Sequence[float] | str | Path,
    path: str | Path | np.ndarray | Sequence[float],
) -> Path:
    """Write 1D diffusion ``bvals`` (s/mm^2) to a space-delimited ``.bval`` file."""
    return _write_numeric_schedule(bvals, path)


def read_te_values(path: str | Path) -> np.ndarray:
    """Read a whitespace- or comma-delimited ``.te`` echo-time file into a 1D ``float64`` array (ms)."""
    return _read_numeric_schedule(path, "TE")


def write_te_values(
    te_values: np.ndarray | Sequence[float] | str | Path,
    path: str | Path | np.ndarray | Sequence[float],
) -> Path:
    """Write 1D echo times ``te_values`` (ms) to a space-delimited ``.te`` file."""
    return _write_numeric_schedule(te_values, path)


@dataclass
class VolumeFitResult(Mapping[str, Any]):
    """Structured parameter map container returned by :func:`fit_volume`."""

    D: np.ndarray
    D_star: np.ndarray
    f: np.ndarray
    S0: np.ndarray
    railed_mask: np.ndarray
    lower_mask: np.ndarray
    upper_mask: np.ndarray
    mode: Literal["1d", "2d"] = "1d"
    method: str = "segmented"
    T2_p: np.ndarray | None = None
    T2_t: np.ndarray | None = None
    rss: np.ndarray | None = None
    success: np.ndarray | None = None
    railed_params: dict[str, np.ndarray] | None = None
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    affine: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    n_voxels_fitted: int = 0

    @property
    def frac_railed(self) -> float:
        """Fraction of finite fitted voxels whose :math:`D^*` estimate railed."""
        finite = np.isfinite(self.D_star)
        if not np.any(finite):
            return float("nan")
        return float(np.mean(self.railed_mask[finite]))

    def keys(self) -> list[str]:  # type: ignore[override]
        base = ["D", "D_star", "f"]
        if self.mode == "2d" and self.T2_p is not None and self.T2_t is not None:
            base.extend(["T2_p", "T2_t"])
        base.extend(["S0", "railed_mask"])
        return base

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __getitem__(self, key: str) -> Any:
        mapping: dict[str, Any] = {
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
            "T2_p": self.T2_p,
            "T2_t": self.T2_t,
            "rss": self.rss,
            "success": self.success,
            "railed_params": self.railed_params,
            "bounds": self.bounds,
            "affine": self.affine,
            "mode": self.mode,
            "method": self.method,
            "frac_railed": self.frac_railed,
            "n_voxels_fitted": self.n_voxels_fitted,
        }
        if key in mapping and mapping[key] is not None:
            return mapping[key]
        raise KeyError(f"Key {key!r} not available in VolumeFitResult (mode={self.mode!r}).")

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary of active spatial maps and metadata."""
        out: dict[str, Any] = {
            "D": self.D,
            "D_star": self.D_star,
            "f": self.f,
            "S0": self.S0,
            "railed_mask": self.railed_mask,
            "lower_mask": self.lower_mask,
            "upper_mask": self.upper_mask,
            "mode": self.mode,
            "method": self.method,
            "bounds": self.bounds,
            "affine": self.affine,
            "frac_railed": self.frac_railed,
            "n_voxels_fitted": self.n_voxels_fitted,
        }
        if self.T2_p is not None:
            out["T2_p"] = self.T2_p
        if self.T2_t is not None:
            out["T2_t"] = self.T2_t
        if self.rss is not None:
            out["rss"] = self.rss
        if self.success is not None:
            out["success"] = self.success
        if self.railed_params is not None:
            out["railed_params"] = self.railed_params
        return out


def _fit_1d_chunk_worker(
    args: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        str,
        float,
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
) -> np.ndarray:
    """Top-level picklable worker for 1D IVIM voxel chunks (returns shape ``(K, 4)``)."""
    sig_chunk, bvals, min_b_mask, method, b_thresh, bounds_D, bounds_D_star, bounds_f = args
    k = sig_chunk.shape[0]
    out = np.full((k, 4), np.nan, dtype=np.float64)
    for i in range(k):
        vox = sig_chunk[i]
        out[i, 3] = float(np.mean(vox[min_b_mask]))
        if method == "segmented":
            d_v, ds_v, f_v = fit_segmented_voxel(
                bvals=bvals,
                signal=vox,
                b_thresh=b_thresh,
                bounds_D=bounds_D,
                bounds_f=bounds_f,
                bounds_D_star=bounds_D_star,
            )
        else:
            d_v, ds_v, f_v = fit_nlls_voxel(
                bvals=bvals,
                signal=vox,
                bounds_D=bounds_D,
                bounds_D_star=bounds_D_star,
                bounds_f=bounds_f,
                b_thresh=b_thresh,
            )
        out[i, 0] = d_v
        out[i, 1] = ds_v
        out[i, 2] = f_v
    return out


def _fit_2d_chunk_worker(
    args: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-level picklable worker for 2D ``(b, TE)`` T2-IVIM voxel chunks."""
    sig_chunk, b_flat, te_flat, lo5, hi5, b_thresh, max_nfev, n_starts = args
    k = sig_chunk.shape[0]
    params = np.full((k, 6), np.nan, dtype=np.float64)
    rss = np.full(k, np.nan, dtype=np.float64)
    ok = np.zeros(k, dtype=bool)
    for i in range(k):
        p6, r_v, ok_v = fit_t2_ivim_voxel(
            b_flat=b_flat,
            te_flat=te_flat,
            signal_flat=sig_chunk[i],
            lo5=lo5,
            hi5=hi5,
            b_thresh=b_thresh,
            max_nfev=max_nfev,
            n_starts=n_starts,
        )
        params[i] = p6
        rss[i] = r_v
        ok[i] = ok_v
    return params, rss, ok


def _coerce_volume_array(
    obj: np.ndarray | NiftiVolume | str | Path | Any,
    use_nibabel: bool | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(obj, NiftiVolume):
        return np.asarray(obj.data, dtype=np.float64), np.asarray(obj.affine, dtype=np.float64)
    if hasattr(obj, "get_fdata"):
        aff = getattr(obj, "affine", None)
        return (
            np.asarray(obj.get_fdata(), dtype=np.float64),
            (None if aff is None else np.asarray(aff, dtype=np.float64)),
        )
    if isinstance(obj, (str, Path)):
        vol = read_nifti(obj, use_nibabel=use_nibabel, dtype=np.float64)
        return vol.data, vol.affine
    return np.asarray(obj, dtype=np.float64), None


def fit_volume(
    dwi: np.ndarray | NiftiVolume | str | Path | Any,
    bvalues: np.ndarray | Sequence[float],
    mask: np.ndarray | NiftiVolume | str | Path | Any | None = None,
    te_values: np.ndarray | Sequence[float] | None = None,
    *,
    mode: Literal["auto", "1d", "2d"] = "auto",
    method: Literal["segmented", "nlls"] = "segmented",
    n_jobs: int = 1,
    chunk_size: int = 256,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    b_thresh: float = DEFAULT_B_THRESH,
    rail_tol: float = DEFAULT_RAIL_TOL,
    grid: bool | None = None,
    max_nfev: int = 350,
    n_starts: int = 3,
    affine: np.ndarray | None = None,
    use_nibabel: bool | None = None,
) -> VolumeFitResult:
    """Fit 1D ``S(b)`` or 2D ``S(b, TE)`` IVIM parameter maps across a 3D/4D DWI volume.

    Executes deterministic chunked fitting via :class:`concurrent.futures.ProcessPoolExecutor`
    when ``n_jobs != 1``, guaranteeing exact bit-for-bit parity between ``n_jobs=1``
    and ``n_jobs>=2``.

    Parameters
    ----------
    dwi : np.ndarray, NiftiVolume, nibabel.Nifti1Image, str, or Path
        4D volumetric DWI array of shape ``(X, Y, Z, M)`` (also supports 3D
        slice stacks ``(X, Y, M)`` or 5D cartesian grids ``(X, Y, Z, N_b, N_te)``).
    bvalues : np.ndarray or Sequence[float]
        Diffusion :math:`b`-values (:math:`\\text{s}/\\text{mm}^2`).
    mask : np.ndarray, NiftiVolume, str, or Path, optional
        Boolean or binary spatial mask matching the spatial dimensions of ``dwi``.
    te_values : np.ndarray or Sequence[float], optional
        Echo times (:math:`\\text{ms}`) enabling 2D relaxation-compensated fitting.
    mode : {"auto", "1d", "2d"}, default="auto"
        ``"auto"`` selects ``"2d"`` when ``te_values`` is provided and ``"1d"`` otherwise.
    method : {"segmented", "nlls"}, default="segmented"
        1D IVIM fitting algorithm when ``mode == "1d"``.
    n_jobs : int, default=1
        Number of worker processes (``1`` runs sequentially in-process; ``-1``
        uses ``os.cpu_count()``).
    chunk_size : int, default=256
        Number of active voxels dispatched per worker task.
    bounds : dict, optional
        Parameter box bounds for ``D``, ``D_star``, ``f`` (and ``T2_p``, ``T2_t`` in 2D mode).
    b_thresh : float, default=200.0
        High-:math:`b` threshold (:math:`\\text{s}/\\text{mm}^2`).
    rail_tol : float, default=1e-2
        Relative tolerance for boundary-railing classification.
    grid : bool or None, optional
        Cartesian ``(b, TE)`` schedule flag for 2D mode.
    affine : np.ndarray, optional
        Override ``(4, 4)`` spatial affine matrix.
    use_nibabel : bool or None, optional
        Backend selector when loading file paths.

    Returns
    -------
    VolumeFitResult
        Mapping-compatible container of 3D spatial parameter maps.
    """
    if chunk_size < 1:
        raise ValueError(f"`chunk_size` must be >= 1; got {chunk_size}.")
    if mode not in ("auto", "1d", "2d"):
        raise ValueError(f"Unsupported `mode` {mode!r}; expected 'auto', '1d', or '2d'.")

    resolved_mode: Literal["1d", "2d"]
    if mode == "auto":
        resolved_mode = "2d" if te_values is not None else "1d"
    else:
        resolved_mode = mode
    if resolved_mode == "2d" and te_values is None:
        raise ValueError("`te_values` must be provided when `mode='2d'`.")

    dwi_arr, loaded_affine = _coerce_volume_array(dwi, use_nibabel=use_nibabel)
    vol_affine = (
        np.asarray(affine, dtype=np.float64)
        if affine is not None
        else (loaded_affine if loaded_affine is not None else np.eye(4, dtype=np.float64))
    )

    if dwi_arr.ndim < 2:
        raise ValueError(
            f"`dwi` must have at least 1 spatial axis plus acquisition axis; got shape {dwi_arr.shape}."
        )

    workers = (os.cpu_count() or 1) if n_jobs < 0 else max(int(n_jobs), 1)

    if resolved_mode == "1d":
        if method not in ("segmented", "nlls"):
            raise ValueError(f"Unknown 1D `method` {method!r}; expected 'segmented' or 'nlls'.")
        b_arr = np.asarray(bvalues, dtype=np.float64)
        if b_arr.ndim != 1 or b_arr.size < 3:
            raise ValueError(f"1D `bvalues` must have length >= 3; got shape {b_arr.shape}.")
        if dwi_arr.shape[-1] != b_arr.size:
            raise ValueError(
                f"Last axis of `dwi` ({dwi_arr.shape[-1]}) must match len(bvalues) ({b_arr.size})."
            )

        spatial_shape = dwi_arr.shape[:-1]
        n_vox = int(np.prod(spatial_shape))
        flat_sig = np.ascontiguousarray(dwi_arr.reshape(n_vox, b_arr.size), dtype=np.float64)

        b_dict_full = _normalize_bounds(bounds)
        b_d = b_dict_full["D"]
        b_ds = b_dict_full["D_star"]
        b_f = b_dict_full["f"]
        bounds_1d = {"D": b_d, "D_star": b_ds, "f": b_f}

        if mask is not None:
            mask_arr, _ = _coerce_volume_array(mask, use_nibabel=use_nibabel)
            if mask_arr.shape != spatial_shape:
                raise ValueError(
                    f"`mask` shape {mask_arr.shape} must match spatial shape {spatial_shape}."
                )
            flat_active = (mask_arr.reshape(n_vox) > 0) & np.all(np.isfinite(flat_sig), axis=1)
        else:
            flat_active = np.all(np.isfinite(flat_sig), axis=1) & (np.max(flat_sig, axis=1) > 0.0)

        active_idx = np.flatnonzero(flat_active)
        min_b_mask = np.isclose(b_arr, float(np.min(b_arr)), atol=1e-6)

        tasks_1d = [
            (
                flat_sig[active_idx[start : start + chunk_size]],
                b_arr,
                min_b_mask,
                method,
                float(b_thresh),
                b_d,
                b_ds,
                b_f,
            )
            for start in range(0, active_idx.size, chunk_size)
        ]

        if workers == 1 or len(tasks_1d) <= 1:
            chunk_results_1d = [_fit_1d_chunk_worker(t) for t in tasks_1d]
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks_1d)), mp_context=ctx) as ex:
                chunk_results_1d = list(ex.map(_fit_1d_chunk_worker, tasks_1d))

        full_1d = np.full((n_vox, 4), np.nan, dtype=np.float64)
        if chunk_results_1d:
            full_1d[active_idx] = np.vstack(chunk_results_1d)

        d_map = full_1d[:, 0].reshape(spatial_shape)
        ds_map = full_1d[:, 1].reshape(spatial_shape)
        f_map = full_1d[:, 2].reshape(spatial_shape)
        s0_map = full_1d[:, 3].reshape(spatial_shape)
        rail_info = classify_railing(ds_map, bounds_D_star=b_ds, rel_tol=rail_tol)

        return VolumeFitResult(
            D=d_map,
            D_star=ds_map,
            f=f_map,
            S0=s0_map,
            railed_mask=np.asarray(rail_info["railed_mask"], dtype=bool),
            lower_mask=np.asarray(rail_info["lower_mask"], dtype=bool),
            upper_mask=np.asarray(rail_info["upper_mask"], dtype=bool),
            mode="1d",
            method=method,
            bounds=bounds_1d,
            affine=vol_affine,
            n_voxels_fitted=int(active_idx.size),
        )

    # 2D (b, TE) volumetric fitting
    assert te_values is not None
    b_flat, te_flat, acq_shape, is_grid = _resolve_schedule(bvalues, te_values, grid=grid)
    m_pts = int(b_flat.size)
    if m_pts < 6:
        raise ValueError(f"2D T2-IVIM requires at least 6 (b, TE) points; got {m_pts}.")

    if is_grid and dwi_arr.ndim >= 3 and dwi_arr.shape[-2:] == acq_shape:
        spatial_shape = dwi_arr.shape[:-2]
    elif dwi_arr.shape[-1] == m_pts:
        spatial_shape = dwi_arr.shape[:-1]
    else:
        raise ValueError(
            f"`dwi` shape {dwi_arr.shape} does not match 2D schedule length {m_pts} "
            f"(grid shape {acq_shape})."
        )

    n_vox = int(np.prod(spatial_shape))
    flat_sig = np.ascontiguousarray(dwi_arr.reshape(n_vox, m_pts), dtype=np.float64)

    bounds_2d = _normalize_bounds(bounds)
    lo5 = np.array([bounds_2d[k][0] for k in PARAM_NAMES_5], dtype=np.float64)
    hi5 = np.array([bounds_2d[k][1] for k in PARAM_NAMES_5], dtype=np.float64)

    if mask is not None:
        mask_arr, _ = _coerce_volume_array(mask, use_nibabel=use_nibabel)
        if mask_arr.shape != spatial_shape:
            raise ValueError(
                f"`mask` shape {mask_arr.shape} must match spatial shape {spatial_shape}."
            )
        flat_active = (mask_arr.reshape(n_vox) > 0) & np.all(np.isfinite(flat_sig), axis=1)
    else:
        flat_active = np.all(np.isfinite(flat_sig), axis=1) & (np.max(flat_sig, axis=1) > 0.0)

    active_idx = np.flatnonzero(flat_active)
    tasks_2d = [
        (
            flat_sig[active_idx[start : start + chunk_size]],
            b_flat,
            te_flat,
            lo5,
            hi5,
            float(b_thresh),
            int(max_nfev),
            int(n_starts),
        )
        for start in range(0, active_idx.size, chunk_size)
    ]

    if workers == 1 or len(tasks_2d) <= 1:
        chunk_results_2d = [_fit_2d_chunk_worker(t) for t in tasks_2d]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks_2d)), mp_context=ctx) as ex:
            chunk_results_2d = list(ex.map(_fit_2d_chunk_worker, tasks_2d))

    full_p = np.full((n_vox, 6), np.nan, dtype=np.float64)
    full_rss = np.full(n_vox, np.nan, dtype=np.float64)
    full_ok = np.zeros(n_vox, dtype=bool)
    if chunk_results_2d:
        full_p[active_idx] = np.vstack([c[0] for c in chunk_results_2d])
        full_rss[active_idx] = np.concatenate([c[1] for c in chunk_results_2d])
        full_ok[active_idx] = np.concatenate([c[2] for c in chunk_results_2d])

    d_map = full_p[:, 0].reshape(spatial_shape)
    ds_map = full_p[:, 1].reshape(spatial_shape)
    f_map = full_p[:, 2].reshape(spatial_shape)
    t2p_map = full_p[:, 3].reshape(spatial_shape)
    t2t_map = full_p[:, 4].reshape(spatial_shape)
    s0_map = full_p[:, 5].reshape(spatial_shape)
    rss_map = full_rss.reshape(spatial_shape)
    ok_map = full_ok.reshape(spatial_shape)

    maps_dict = {
        "D": d_map,
        "D_star": ds_map,
        "f": f_map,
        "T2_p": t2p_map,
        "T2_t": t2t_map,
    }
    railed_params = {
        name: np.asarray(
            classify_railing(maps_dict[name], bounds_D_star=bounds_2d[name], rel_tol=rail_tol)[
                "railed_mask"
            ],
            dtype=bool,
        )
        for name in PARAM_NAMES_5
    }
    ds_info = classify_railing(ds_map, bounds_D_star=bounds_2d["D_star"], rel_tol=rail_tol)

    return VolumeFitResult(
        D=d_map,
        D_star=ds_map,
        f=f_map,
        S0=s0_map,
        railed_mask=np.asarray(ds_info["railed_mask"], dtype=bool),
        lower_mask=np.asarray(ds_info["lower_mask"], dtype=bool),
        upper_mask=np.asarray(ds_info["upper_mask"], dtype=bool),
        mode="2d",
        method="t2_nlls",
        T2_p=t2p_map,
        T2_t=t2t_map,
        rss=rss_map,
        success=ok_map,
        railed_params=railed_params,
        bounds=bounds_2d,
        affine=vol_affine,
        n_voxels_fitted=int(active_idx.size),
    )


def _finite_median(arr: np.ndarray | None) -> float | None:
    if arr is None:
        return None
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def fit_volume_files(
    dwi_path: str | Path,
    bval_path: str | Path,
    out_dir: str | Path,
    mask_path: str | Path | None = None,
    te_path: str | Path | None = None,
    n_jobs: int = 1,
    *,
    mode: Literal["auto", "1d", "2d"] = "auto",
    method: Literal["segmented", "nlls"] = "segmented",
    chunk_size: int = 256,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    b_thresh: float = DEFAULT_B_THRESH,
    rail_tol: float = DEFAULT_RAIL_TOL,
    use_nibabel: bool | None = None,
) -> dict[str, Any]:
    """Load NIfTI + schedule files, fit 1D or 2D IVIM maps, and write NIfTI maps + ``summary.json``.

    Writes ``D.nii.gz``, ``D_star.nii.gz``, ``f.nii.gz``, ``S0.nii.gz``,
    ``railed_mask.nii.gz`` (plus ``T2_p.nii.gz`` and ``T2_t.nii.gz`` when
    ``te_path`` is provided), and ``summary.json`` into ``out_dir``.
    """
    dwi_vol = read_nifti(dwi_path, use_nibabel=use_nibabel, dtype=np.float64)
    bvals = read_bvals(bval_path)
    te_vals = read_te_values(te_path) if te_path is not None else None
    mask_arr = (
        read_nifti(mask_path, use_nibabel=use_nibabel, dtype=np.float64).data
        if mask_path is not None
        else None
    )

    res = fit_volume(
        dwi=dwi_vol,
        bvalues=bvals,
        mask=mask_arr,
        te_values=te_vals,
        mode=mode,
        method=method,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        bounds=bounds,
        b_thresh=b_thresh,
        rail_tol=rail_tol,
        use_nibabel=use_nibabel,
    )

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, str] = {}
    float_maps: list[tuple[str, np.ndarray]] = [
        ("D", res.D),
        ("D_star", res.D_star),
        ("f", res.f),
        ("S0", res.S0),
    ]
    if res.mode == "2d" and res.T2_p is not None and res.T2_t is not None:
        float_maps.extend([("T2_p", res.T2_p), ("T2_t", res.T2_t)])

    for map_name, map_arr in float_maps:
        target = out_p / f"{map_name}.nii.gz"
        write_nifti(
            map_arr,
            target,
            affine=res.affine,
            dtype=np.float32,
            use_nibabel=use_nibabel,
        )
        written_files[map_name] = str(target)

    rail_target = out_p / "railed_mask.nii.gz"
    write_nifti(
        res.railed_mask.astype(np.uint8),
        rail_target,
        affine=res.affine,
        dtype=np.uint8,
        use_nibabel=use_nibabel,
    )
    written_files["railed_mask"] = str(rail_target)

    finite_ds = np.isfinite(res.D_star)
    n_railed = int(np.sum(res.railed_mask[finite_ds])) if np.any(finite_ds) else 0
    frac_railed = None if not np.isfinite(res.frac_railed) else float(res.frac_railed)

    summary: dict[str, Any] = {
        "package": "sextant-core",
        "version": "1.1.0",
        "mode": res.mode,
        "method": res.method,
        "spatial_shape": list(res.D.shape),
        "n_acquisitions": int(dwi_vol.data.shape[-1]),
        "n_voxels_total": int(res.D.size),
        "n_voxels_fitted": int(res.n_voxels_fitted),
        "n_railed": n_railed,
        "frac_railed": frac_railed,
        "n_jobs": int(n_jobs),
        "bounds": {k: [float(v[0]), float(v[1])] for k, v in res.bounds.items()},
        "medians": {
            "D": _finite_median(res.D),
            "D_star": _finite_median(res.D_star),
            "f": _finite_median(res.f),
            "S0": _finite_median(res.S0),
            "T2_p": _finite_median(res.T2_p),
            "T2_t": _finite_median(res.T2_t),
        },
        "outputs": {k: Path(v).name for k, v in written_files.items()},
    }
    summary_path = out_p / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    written_files["summary"] = str(summary_path)

    return {
        "summary": summary,
        "files": written_files,
        "result": res,
    }
