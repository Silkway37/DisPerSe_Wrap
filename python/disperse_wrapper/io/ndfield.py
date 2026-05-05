"""NDfield format I/O for DisPerSE.

DisPerSE's native input/output format is "NDfield".
This module provides writers for:
 - point-cloud NDfield (for ``mse`` point-catalog input)
 - grid NDfield (for ``mse`` scalar-field input)

References
----------
https://www2.iap.fr/users/sousbie/web/html/indexd41d.html?category/NDfield-format
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# ASCII NDfield (text header + separate binary .dat) – simplest portable format
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    np.dtype("float32"): "FLOAT",
    np.dtype("float64"): "DOUBLE",
    np.dtype("int32"): "INT",
    np.dtype("int64"): "LONG",
}

_DTYPE_REVERSE = {v: k for k, v in _DTYPE_MAP.items()}


def write_ndfield(
    data: np.ndarray,
    path: str | Path,
    bbox_min: np.ndarray | None = None,
    bbox_max: np.ndarray | None = None,
) -> Path:
    """Write a numpy array to DisPerSE NDfield (binary) format.

    For a 3-D scalar grid ``data`` should have shape ``(Nx, Ny, Nz)``.
    For a point cloud the array should have shape ``(N, 3)`` – this is stored
    as a 2-D NDfield of shape ``(N, 3)``.

    Parameters
    ----------
    data:
        Array to write.
    path:
        Destination path (the file extension ``.NDfield`` will be appended if
        absent).
    bbox_min, bbox_max:
        Bounding-box corners.  Derived from data extents when omitted.

    Returns
    -------
    Path
        Path to the written ``.NDfield`` file.
    """
    path = Path(path)
    if path.suffix.lower() != ".ndfield":
        path = path.with_suffix(".NDfield")

    data = np.asarray(data)
    ndims = data.ndim
    dims = list(data.shape)

    dtype = data.dtype
    if dtype not in _DTYPE_MAP:
        data = data.astype(np.float32)
        dtype = data.dtype

    dtype_str = _DTYPE_MAP[dtype]

    if bbox_min is None:
        bbox_min = np.zeros(ndims, dtype=np.float64)
    if bbox_max is None:
        bbox_max = np.array(dims, dtype=np.float64)

    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)

    # ------------------------------------------------------------------
    # Write binary NDfield (Fortran-record wrapped, little-endian)
    # Format:  NDFIELDMAGIC + header record + data record
    # DisPerSE reads Fortran-style records: [int32 size][data][int32 size]
    # ------------------------------------------------------------------
    _write_binary_ndfield(path, data, ndims, dims, bbox_min, bbox_max, dtype_str)
    return path


def _write_binary_ndfield(
    path: Path,
    data: np.ndarray,
    ndims: int,
    dims: list[int],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    dtype_str: str,
) -> None:
    """Write a binary NDfield using the format that ``mse`` expects.

    The binary layout is:
    - magic record  : ``b"NDFIELD"``  (Fortran record)
    - header record :
        ``ndims`` (int32),
        ``dims[0..ndims-1]`` (int32 × ndims),
        ``bbox_min[0..ndims-1]`` (float64 × ndims),
        ``bbox_max[0..ndims-1]`` (float64 × ndims),
        ``dtype`` (null-padded 16-byte string)
    - data record   : raw data bytes in C order
    """
    def _write_record(f, payload: bytes) -> None:
        size = len(payload)
        f.write(struct.pack("<i", size))
        f.write(payload)
        f.write(struct.pack("<i", size))

    magic = b"NDFIELD"
    header_payload = struct.pack("<i", ndims)
    for d in dims:
        header_payload += struct.pack("<i", d)
    for v in bbox_min:
        header_payload += struct.pack("<d", float(v))
    for v in bbox_max:
        header_payload += struct.pack("<d", float(v))
    dtype_bytes = dtype_str.encode("ascii").ljust(16, b"\x00")
    header_payload += dtype_bytes

    data_payload = data.astype(data.dtype, order="C").tobytes()

    with open(path, "wb") as f:
        _write_record(f, magic)
        _write_record(f, header_payload)
        _write_record(f, data_payload)


def write_points_ascii(
    points: np.ndarray,
    path: str | Path,
) -> Path:
    """Write an (N, 3) point array to a simple whitespace-delimited ASCII file.

    This is an alternative input to ``mse`` when used with catalog-mode
    (``mse input.txt -smooth 3 …``).  Each line contains ``x y z``.

    Parameters
    ----------
    points:
        Array of shape ``(N, 3)``.
    path:
        Output file path.

    Returns
    -------
    Path
    """
    path = Path(path)
    np.savetxt(str(path), points, fmt="%.8g")
    return path
