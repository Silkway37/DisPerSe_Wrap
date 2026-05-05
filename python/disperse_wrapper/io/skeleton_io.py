"""Skeleton file I/O for DisPerSE.

Supported formats
-----------------
* ``.NDskl``          – DisPerSE native binary skeleton  (read)
* ``.a.NDskl``        – DisPerSE ASCII skeleton produced by ``skelconv -to a.NDskl`` (read/write)
* ``.json``           – JSON  (read/write)
* ``.npz``            – NumPy compressed archive  (read/write)
* ``.hdf5`` / ``.h5`` – HDF5  (read/write, requires ``h5py``)
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..skeleton import Skeleton


# ---------------------------------------------------------------------------
# ASCII NDskl parser (.a.NDskl)
# ---------------------------------------------------------------------------

def _parse_ascii_ndskl(text: str) -> Skeleton:
    """Parse DisPerSE ASCII skeleton format produced by skelconv -to a.NDskl."""
    from ..skeleton import Arc, CriticalPoint, Skeleton

    lines = [ln.rstrip() for ln in text.splitlines()]
    idx = 0

    def _next() -> str:
        nonlocal idx
        while idx < len(lines):
            ln = lines[idx].strip()
            idx += 1
            if ln and not ln.startswith("#"):
                return ln
        return ""

    # --- header ---
    header = _next()
    if not header.upper().startswith("NDSKEL"):
        raise ValueError(f"Expected 'NDSKEL' header, got: {header!r}")

    ndims = 3
    n_nodes = 0
    n_arcs = 0
    bbox_min: np.ndarray | None = None
    bbox_max: np.ndarray | None = None

    # Read key-value header lines until we hit "CRITICAL POINTS" or data section
    while True:
        ln = _next()
        if not ln:
            break
        upper = ln.upper()
        if upper.startswith("NDIMS"):
            ndims = int(ln.split()[-1])
        elif upper.startswith("NVERTEX"):
            n_nodes = int(ln.split()[-1])
        elif upper.startswith("NFILAMENT"):
            n_arcs = int(ln.split()[-1])
        elif upper.startswith("BBOX") or upper.startswith("BOUNDING"):
            parts = ln.split()[1:]
            coords = [float(v) for v in parts]
            half = len(coords) // 2
            bbox_min = np.array(coords[:half])
            bbox_max = np.array(coords[half:])
        elif "CRITICAL POINTS" in upper:
            break

    # --- critical points ---
    critical_points: list[CriticalPoint] = []
    for i in range(n_nodes):
        ln = _next()
        parts = ln.split()
        # Format: x y z value pair_id type [flags] [narc arc1 arc2 ...]
        pos = np.array([float(parts[j]) for j in range(ndims)])
        val = float(parts[ndims]) if len(parts) > ndims else 0.0
        pair_id = int(parts[ndims + 1]) if len(parts) > ndims + 1 else -1
        cp_type = int(parts[ndims + 2]) if len(parts) > ndims + 2 else -1
        critical_points.append(
            CriticalPoint(
                id=i,
                position=pos,
                cp_type=cp_type,
                value=val,
                pair_id=pair_id,
            )
        )

    # skip "FILAMENTS" header line if present
    while True:
        ln = _next()
        if not ln:
            break
        if "FILAMENT" in ln.upper():
            break
        # maybe we already consumed it
        idx -= 1
        break

    # --- arcs / filaments ---
    arcs: list[Arc] = []
    for i in range(n_arcs):
        header_ln = _next()
        hparts = header_ln.split()
        cp1 = int(hparts[0])
        cp2 = int(hparts[1])
        n_samp = int(hparts[2]) if len(hparts) > 2 else 0

        points_list: list[np.ndarray] = []
        for _ in range(n_samp):
            pt_ln = _next()
            coords = [float(v) for v in pt_ln.split()[:ndims]]
            points_list.append(np.array(coords))

        arcs.append(
            Arc(
                id=i,
                node_ids=(cp1, cp2),
                points=np.array(points_list) if points_list else np.empty((0, ndims)),
            )
        )

    return Skeleton(
        ndims=ndims,
        critical_points=critical_points,
        arcs=arcs,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


# ---------------------------------------------------------------------------
# Binary NDskl parser (.NDskl)
# ---------------------------------------------------------------------------

def _read_fortran_record(f) -> bytes:
    """Read one Fortran-style record (int32 size, data, int32 size)."""
    raw = f.read(4)
    if not raw:
        raise EOFError("Unexpected end of file reading record size")
    (size,) = struct.unpack("<i", raw)
    data = f.read(size)
    f.read(4)  # trailing size
    return data


def _parse_binary_ndskl(path: Path) -> Skeleton:
    """Parse DisPerSE native binary .NDskl format."""
    from ..skeleton import Arc, CriticalPoint, Skeleton

    with open(path, "rb") as f:
        # Magic record
        magic = _read_fortran_record(f)
        if b"NDSKEL" not in magic:
            raise ValueError(
                f"File does not start with NDSKEL magic: {magic[:20]!r}"
            )

        # Header record
        header_data = _read_fortran_record(f)
        hf = 0

        def _read_int() -> int:
            nonlocal hf
            (v,) = struct.unpack_from("<i", header_data, hf)
            hf += 4
            return v

        def _read_float() -> float:
            nonlocal hf
            (v,) = struct.unpack_from("<f", header_data, hf)
            hf += 4
            return v

        def _read_double() -> float:
            nonlocal hf
            (v,) = struct.unpack_from("<d", header_data, hf)
            hf += 8
            return v

        ndims = _read_int()
        comment = header_data[hf : hf + 80].decode("latin-1").rstrip("\x00").strip()
        hf += 80

        bbox_min = np.array([_read_double() for _ in range(ndims)])
        bbox_max = np.array([_read_double() for _ in range(ndims)])
        n_nodes = _read_int()
        n_arcs = _read_int()

        # Critical points
        critical_points: list[CriticalPoint] = []
        cp_record = _read_fortran_record(f)
        cf = 0

        def _cp_int() -> int:
            nonlocal cf
            (v,) = struct.unpack_from("<i", cp_record, cf)
            cf += 4
            return v

        def _cp_float() -> float:
            nonlocal cf
            (v,) = struct.unpack_from("<f", cp_record, cf)
            cf += 4
            return v

        for i in range(n_nodes):
            cp_type = _cp_int()
            is_boundary = _cp_int()
            pos = np.array([_cp_float() for _ in range(ndims)])
            value = _cp_float()
            _cp_float()  # pair_value (not used directly, advances offset)
            pair_id = _cp_int()
            n_arcs_cp = _cp_int()
            arc_ids = [_cp_int() for _ in range(n_arcs_cp)]

            critical_points.append(
                CriticalPoint(
                    id=i,
                    position=pos,
                    cp_type=cp_type,
                    value=value,
                    pair_id=pair_id,
                    is_boundary=bool(is_boundary),
                    arc_ids=arc_ids,
                )
            )

        # Arcs
        arcs: list[Arc] = []
        arc_record = _read_fortran_record(f)
        af = 0

        def _arc_int() -> int:
            nonlocal af
            (v,) = struct.unpack_from("<i", arc_record, af)
            af += 4
            return v

        def _arc_float() -> float:
            nonlocal af
            (v,) = struct.unpack_from("<f", arc_record, af)
            af += 4
            return v

        for i in range(n_arcs):
            cp1 = _arc_int()
            cp2 = _arc_int()
            n_samp = _arc_int()
            pts = []
            for _ in range(n_samp):
                pts.append([_arc_float() for _ in range(ndims)])
            arcs.append(
                Arc(
                    id=i,
                    node_ids=(cp1, cp2),
                    points=np.array(pts) if pts else np.empty((0, ndims)),
                )
            )

    return Skeleton(
        ndims=ndims,
        critical_points=critical_points,
        arcs=arcs,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        comment=comment,
    )


# ---------------------------------------------------------------------------
# Public load / save
# ---------------------------------------------------------------------------

def _is_ascii_ndskl(path: Path) -> bool:
    """Return True if *path* looks like an ASCII NDSKEL file (text, not binary)."""
    try:
        header = path.read_bytes()[:16]
        return header.startswith(b"NDSKEL")
    except OSError:
        return False


def load_skeleton_file(path: Path) -> Skeleton:
    """Load a skeleton from *path*, auto-detecting the format by extension.

    For ``.NDskl`` files, the parser first checks whether the content is
    ASCII (produced by ``skelconv -to a.NDskl`` or a compatible tool) and
    falls back to the binary parser if not.

    Supported extensions: ``.a.NDskl``, ``.NDskl``, ``.json``, ``.npz``,
    ``.hdf5``, ``.h5``.
    """
    path = Path(path)
    name = path.name.lower()

    if name.endswith(".a.ndskl"):
        return _parse_ascii_ndskl(path.read_text(encoding="utf-8", errors="replace"))
    if name.endswith(".ndskl"):
        # Sniff: if first bytes are printable ASCII "NDSKEL", treat as ASCII
        if _is_ascii_ndskl(path):
            return _parse_ascii_ndskl(path.read_text(encoding="utf-8", errors="replace"))
        return _parse_binary_ndskl(path)
    if name.endswith(".json"):
        return _load_json(path)
    if name.endswith(".npz"):
        return _load_npz(path)
    if name.endswith((".hdf5", ".h5")):
        return _load_hdf5(path)
    raise ValueError(
        f"Unrecognised skeleton file extension for '{path}'.  "
        "Supported: .a.NDskl, .NDskl, .json, .npz, .hdf5, .h5"
    )


def save_skeleton_file(skeleton: Skeleton, path: Path, fmt: str) -> Path:
    """Save *skeleton* to *path* in the given *fmt*.

    Parameters
    ----------
    skeleton : Skeleton
    path : Path
    fmt : ``"json"``, ``"npz"``, or ``"hdf5"``
    """
    path = Path(path)
    fmt = fmt.lower().lstrip(".")
    if fmt == "json":
        return _save_json(skeleton, path)
    if fmt == "npz":
        return _save_npz(skeleton, path)
    if fmt in ("hdf5", "h5"):
        return _save_hdf5(skeleton, path)
    raise ValueError(f"Unsupported save format: {fmt!r}.  Choose json, npz, or hdf5.")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def _skeleton_to_dict(sk: Skeleton) -> dict[str, Any]:
    return {
        "ndims": sk.ndims,
        "comment": sk.comment,
        "bbox_min": sk.bbox_min.tolist() if sk.bbox_min is not None else None,
        "bbox_max": sk.bbox_max.tolist() if sk.bbox_max is not None else None,
        "critical_points": [
            {
                "id": cp.id,
                "position": cp.position.tolist(),
                "cp_type": cp.cp_type,
                "value": float(cp.value),
                "pair_id": cp.pair_id,
                "is_boundary": cp.is_boundary,
                "arc_ids": cp.arc_ids,
            }
            for cp in sk.critical_points
        ],
        "arcs": [
            {
                "id": arc.id,
                "node_ids": list(arc.node_ids),
                "points": arc.points.tolist(),
                "persistence": arc.persistence,
                "length": float(arc.length) if arc.length is not None else None,
            }
            for arc in sk.arcs
        ],
    }


def _save_json(skeleton: Skeleton, path: Path) -> Path:
    path = path if path.suffix else path.with_suffix(".json")
    path.write_text(json.dumps(_skeleton_to_dict(skeleton), indent=2), encoding="utf-8")
    return path


def _load_json(path: Path) -> Skeleton:
    from ..skeleton import Arc, CriticalPoint, Skeleton

    d = json.loads(path.read_text(encoding="utf-8"))
    cps = [
        CriticalPoint(
            id=cp["id"],
            position=np.array(cp["position"]),
            cp_type=cp.get("cp_type", -1),
            value=cp.get("value", 0.0),
            pair_id=cp.get("pair_id", -1),
            is_boundary=cp.get("is_boundary", False),
            arc_ids=cp.get("arc_ids", []),
        )
        for cp in d["critical_points"]
    ]
    arcs = [
        Arc(
            id=arc["id"],
            node_ids=tuple(arc["node_ids"]),
            points=np.array(arc["points"]),
            persistence=arc.get("persistence"),
            length=arc.get("length"),
        )
        for arc in d["arcs"]
    ]
    return Skeleton(
        ndims=d.get("ndims", 3),
        critical_points=cps,
        arcs=arcs,
        bbox_min=np.array(d["bbox_min"]) if d.get("bbox_min") else None,
        bbox_max=np.array(d["bbox_max"]) if d.get("bbox_max") else None,
        comment=d.get("comment", ""),
    )


# ---------------------------------------------------------------------------
# NPZ
# ---------------------------------------------------------------------------

def _save_npz(skeleton: Skeleton, path: Path) -> Path:
    path = path if path.suffix else path.with_suffix(".npz")
    d = _skeleton_to_dict(skeleton)

    # Flatten for npz
    cp_pos = np.array([cp["position"] for cp in d["critical_points"]], dtype=np.float64)
    cp_type = np.array([cp["cp_type"] for cp in d["critical_points"]], dtype=np.int32)
    cp_val = np.array([cp["value"] for cp in d["critical_points"]], dtype=np.float64)
    cp_pair = np.array([cp["pair_id"] for cp in d["critical_points"]], dtype=np.int32)
    cp_boundary = np.array([cp["is_boundary"] for cp in d["critical_points"]], dtype=bool)

    arc_node_ids = np.array([a["node_ids"] for a in d["arcs"]], dtype=np.int32)
    arc_lengths = np.array(
        [a["length"] if a["length"] is not None else np.nan for a in d["arcs"]],
        dtype=np.float64,
    )
    # Store arc points as a ragged object array
    arc_pts = np.empty(len(d["arcs"]), dtype=object)
    for i, a in enumerate(d["arcs"]):
        arc_pts[i] = np.array(a["points"], dtype=np.float64)

    meta = {
        "ndims": skeleton.ndims,
        "comment": skeleton.comment or "",
    }

    save_dict: dict[str, Any] = {
        "meta_json": np.bytes_(json.dumps(meta)),
        "cp_positions": cp_pos,
        "cp_types": cp_type,
        "cp_values": cp_val,
        "cp_pair_ids": cp_pair,
        "cp_boundary": cp_boundary,
        "arc_node_ids": arc_node_ids,
        "arc_lengths": arc_lengths,
        "arc_points": arc_pts,
    }
    if skeleton.bbox_min is not None:
        save_dict["bbox_min"] = skeleton.bbox_min
        save_dict["bbox_max"] = skeleton.bbox_max

    np.savez_compressed(str(path), **save_dict)
    return path


def _load_npz(path: Path) -> Skeleton:
    from ..skeleton import Arc, CriticalPoint, Skeleton

    data = np.load(str(path), allow_pickle=True)
    meta = json.loads(data["meta_json"].tobytes())

    n_cp = len(data["cp_positions"])
    cps = [
        CriticalPoint(
            id=i,
            position=data["cp_positions"][i],
            cp_type=int(data["cp_types"][i]),
            value=float(data["cp_values"][i]),
            pair_id=int(data["cp_pair_ids"][i]),
            is_boundary=bool(data["cp_boundary"][i]),
        )
        for i in range(n_cp)
    ]

    n_arc = len(data["arc_node_ids"])
    arcs = [
        Arc(
            id=i,
            node_ids=tuple(data["arc_node_ids"][i].tolist()),
            points=data["arc_points"][i],
            length=float(data["arc_lengths"][i])
            if not np.isnan(data["arc_lengths"][i])
            else None,
        )
        for i in range(n_arc)
    ]

    bbox_min = data["bbox_min"] if "bbox_min" in data else None
    bbox_max = data["bbox_max"] if "bbox_max" in data else None

    return Skeleton(
        ndims=meta.get("ndims", 3),
        critical_points=cps,
        arcs=arcs,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        comment=meta.get("comment", ""),
    )


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------

def _save_hdf5(skeleton: Skeleton, path: Path) -> Path:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 I/O.  "
            "Install it with: pip install disperse-wrapper[io]"
        ) from exc

    path = path if path.suffix else path.with_suffix(".hdf5")
    with h5py.File(path, "w") as f:
        f.attrs["ndims"] = skeleton.ndims
        f.attrs["comment"] = skeleton.comment or ""
        if skeleton.bbox_min is not None:
            f.attrs["bbox_min"] = skeleton.bbox_min
            f.attrs["bbox_max"] = skeleton.bbox_max

        cp_grp = f.create_group("critical_points")
        if skeleton.critical_points:
            cp_grp.create_dataset(
                "positions",
                data=np.array([cp.position for cp in skeleton.critical_points]),
            )
            cp_grp.create_dataset(
                "types",
                data=np.array([cp.cp_type for cp in skeleton.critical_points], dtype=np.int32),
            )
            cp_grp.create_dataset(
                "values",
                data=np.array([cp.value for cp in skeleton.critical_points]),
            )
            cp_grp.create_dataset(
                "pair_ids",
                data=np.array([cp.pair_id for cp in skeleton.critical_points], dtype=np.int32),
            )
            cp_grp.create_dataset(
                "boundary",
                data=np.array([cp.is_boundary for cp in skeleton.critical_points], dtype=bool),
            )

        arc_grp = f.create_group("arcs")
        if skeleton.arcs:
            arc_grp.create_dataset(
                "node_ids",
                data=np.array([arc.node_ids for arc in skeleton.arcs], dtype=np.int32),
            )
            arc_grp.create_dataset(
                "lengths",
                data=np.array(
                    [arc.length if arc.length is not None else np.nan for arc in skeleton.arcs]
                ),
            )
            # Store ragged arc points using variable-length arrays
            dt = h5py.vlen_dtype(np.float64)
            pts_flat = arc_grp.create_dataset("points_flat", (len(skeleton.arcs),), dtype=dt)
            arc_grp.create_dataset(
                "points_npts",
                data=np.array([len(arc.points) for arc in skeleton.arcs], dtype=np.int32),
            )
            for i, arc in enumerate(skeleton.arcs):
                pts_flat[i] = arc.points.flatten()

    return path


def _load_hdf5(path: Path) -> Skeleton:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 I/O.  "
            "Install it with: pip install disperse-wrapper[io]"
        ) from exc

    from ..skeleton import Arc, CriticalPoint, Skeleton

    with h5py.File(path, "r") as f:
        ndims = int(f.attrs.get("ndims", 3))
        comment = str(f.attrs.get("comment", ""))
        bbox_min = np.array(f.attrs["bbox_min"]) if "bbox_min" in f.attrs else None
        bbox_max = np.array(f.attrs["bbox_max"]) if "bbox_max" in f.attrs else None

        cp_grp = f["critical_points"]
        positions = np.array(cp_grp["positions"]) if "positions" in cp_grp else np.empty((0, ndims))
        cp_types = np.array(cp_grp["types"]) if "types" in cp_grp else np.full(len(positions), -1)
        cp_values = np.array(cp_grp["values"]) if "values" in cp_grp else np.zeros(len(positions))
        cp_pair_ids = np.array(cp_grp["pair_ids"]) if "pair_ids" in cp_grp else np.full(len(positions), -1)
        cp_boundary = np.array(cp_grp["boundary"]) if "boundary" in cp_grp else np.zeros(len(positions), dtype=bool)

        cps = [
            CriticalPoint(
                id=i,
                position=positions[i],
                cp_type=int(cp_types[i]),
                value=float(cp_values[i]),
                pair_id=int(cp_pair_ids[i]),
                is_boundary=bool(cp_boundary[i]),
            )
            for i in range(len(positions))
        ]

        arc_grp = f["arcs"]
        arc_node_ids = np.array(arc_grp["node_ids"]) if "node_ids" in arc_grp else np.empty((0, 2), dtype=np.int32)
        arc_lengths = np.array(arc_grp["lengths"]) if "lengths" in arc_grp else np.full(len(arc_node_ids), np.nan)
        pts_flat = arc_grp["points_flat"] if "points_flat" in arc_grp else []
        npts_arr = np.array(arc_grp["points_npts"]) if "points_npts" in arc_grp else np.zeros(len(arc_node_ids), dtype=np.int32)

        arcs = []
        for i in range(len(arc_node_ids)):
            flat = np.array(pts_flat[i])
            n = int(npts_arr[i])
            pts = flat.reshape(n, ndims) if n > 0 else np.empty((0, ndims))
            arcs.append(
                Arc(
                    id=i,
                    node_ids=tuple(arc_node_ids[i].tolist()),
                    points=pts,
                    length=float(arc_lengths[i]) if not np.isnan(arc_lengths[i]) else None,
                )
            )

    return Skeleton(
        ndims=ndims,
        critical_points=cps,
        arcs=arcs,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        comment=comment,
    )
