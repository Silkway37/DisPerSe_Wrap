"""High-level Python API for running DisPerSE.

Main entry points
-----------------
* :func:`run_disperse_points` – run mse on a point-cloud (N×3 array).
* :func:`run_disperse_grid`   – run mse on a scalar density grid.
* :func:`load_skeleton`       – load a skeleton from a file.
* :func:`save_skeleton`       – save a skeleton to JSON / NPZ / HDF5.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import DisperseInputError
from .io import load_skeleton_file, save_skeleton_file, write_ndfield
from .runner import find_binary, run_binary
from .skeleton import Skeleton

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass
class DisperseRunResult:
    """Result returned by :func:`run_disperse_points` or :func:`run_disperse_grid`.

    Attributes
    ----------
    workdir : Path
        Directory where all intermediate and output files reside.
    skeleton_files : list[Path]
        Paths to raw skeleton files produced by DisPerSE (e.g. ``.NDskl``,
        ``.a.NDskl``).
    skeleton : Skeleton | None
        Loaded :class:`~disperse_wrapper.skeleton.Skeleton` object (populated
        when the skeleton could be parsed automatically).
    commands : list[list[str]]
        Ordered list of command-line invocations that were executed.
    parameters : dict[str, Any]
        Key/value record of all run parameters.
    elapsed_seconds : float
        Wall-clock time for the run.
    """

    workdir: Path
    skeleton_files: list[Path] = field(default_factory=list)
    skeleton: Skeleton | None = None
    commands: list[list[str]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def __repr__(self) -> str:
        sk = repr(self.skeleton) if self.skeleton else "None"
        return (
            f"DisperseRunResult(workdir={self.workdir}, "
            f"n_files={len(self.skeleton_files)}, "
            f"skeleton={sk})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise DisperseInputError(
            f"points must be a 2-D array with shape (N, 3), got shape {points.shape}"
        )
    if len(points) < 4:
        raise DisperseInputError("points must contain at least 4 points")
    return points


def _validate_grid(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim != 3:
        raise DisperseInputError(
            f"grid must be a 3-D array with shape (Nx, Ny, Nz), got shape {grid.shape}"
        )
    return grid


def _validate_boxsize(boxsize: float | tuple | list | np.ndarray, ndims: int = 3) -> np.ndarray:
    bs = np.asarray(boxsize, dtype=np.float64).ravel()
    if len(bs) == 1:
        bs = np.repeat(bs, ndims)
    if len(bs) != ndims:
        raise DisperseInputError(
            f"boxsize must be a scalar or a sequence of length {ndims}, got {boxsize!r}"
        )
    if np.any(bs <= 0):
        raise DisperseInputError(f"All boxsize values must be positive, got {boxsize!r}")
    return bs


def _make_workdir(workdir: Path | str | None) -> tuple[Path, bool]:
    """Return (workdir, is_temp)."""
    if workdir is None:
        d = Path(tempfile.mkdtemp(prefix="disperse_"))
        return d, True
    d = Path(workdir)
    d.mkdir(parents=True, exist_ok=True)
    return d, False


def _find_skeleton_files(workdir: Path, stem: str) -> list[Path]:
    """Collect all .NDskl / .a.NDskl files produced under *workdir*."""
    patterns = [
        f"{stem}.NDskl",
        f"{stem}.a.NDskl",
        f"{stem}.up.NDskl",
        f"{stem}.down.NDskl",
    ]
    found = []
    for pat in patterns:
        p = workdir / pat
        if p.exists():
            found.append(p)
    # fallback: any .NDskl in workdir
    if not found:
        found = list(workdir.glob("*.NDskl"))
    return found


def _try_load_skeleton(skeleton_files: list[Path]) -> Skeleton | None:
    """Try to load a skeleton from the first parseable file."""
    # prefer ASCII .a.NDskl
    order = sorted(
        skeleton_files,
        key=lambda p: (0 if p.name.endswith(".a.NDskl") else 1),
    )
    for fp in order:
        try:
            return load_skeleton_file(fp)
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not parse %s: %s", fp, exc)
    return None


def _build_mse_args(
    input_file: Path,
    *,
    periodic: bool,
    persistence: float | None,
    smooth: int | None,
    extra_kwargs: dict[str, Any],
) -> list[str]:
    args: list[str] = [str(input_file)]
    if periodic:
        args.append("-periodic")
    if persistence is not None:
        args += ["-nsig", str(persistence)]
    if smooth is not None:
        args += ["-smooth", str(int(smooth))]
    for k, v in extra_kwargs.items():
        k_flag = k if k.startswith("-") else f"-{k}"
        if v is True:
            args.append(k_flag)
        elif v is not False and v is not None:
            args += [k_flag, str(v)]
    return args


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_disperse_points(
    points: np.ndarray | str | os.PathLike,
    boxsize: float | tuple | list | np.ndarray,
    periodic: bool = True,
    *,
    workdir: str | os.PathLike | None = None,
    persistence: float | None = None,
    smooth: int | None = None,
    binary_dir: str | os.PathLike | None = None,
    keep_intermediate: bool = False,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> DisperseRunResult:
    """Run DisPerSE on a raw point cloud.

    Parameters
    ----------
    points:
        Either a numpy array of shape ``(N, 3)`` (positions) or a path to a
        ``.npy`` file containing such an array.
    boxsize:
        Side length(s) of the simulation box.  Scalar or 3-tuple.
    periodic:
        Enable periodic boundary conditions (maps to ``-periodic`` in mse).
    workdir:
        Directory for all intermediate and output files.  A temporary
        directory is used when *None*.
    persistence:
        Persistence threshold in sigma units (``-nsig``).
    smooth:
        Number of smoothing iterations (``-smooth``).
    binary_dir:
        Directory containing DisPerSE binaries (overrides ``DISPERSE_BIN``
        and PATH).
    keep_intermediate:
        If *False* (default) the working directory is cleaned up after the
        run.  If *True*, everything is kept.
    logger:
        Optional Python logger for subprocess output.
    **kwargs:
        Additional command-line flags passed verbatim to ``mse`` (e.g.
        ``up=True`` → ``-up``).

    Returns
    -------
    DisperseRunResult
    """
    _log = logger or log
    t0 = time.monotonic()

    # --- load points if path given ---
    if isinstance(points, (str, os.PathLike)):
        points_arr = np.load(str(points))
    else:
        points_arr = points  # type: ignore[assignment]
    points_arr = _validate_points(points_arr)

    bs = _validate_boxsize(boxsize)

    wdir, is_temp = _make_workdir(workdir)
    commands: list[list[str]] = []
    stem = "points"

    try:
        # Write NDfield
        nd_path = write_ndfield(
            points_arr,
            wdir / stem,
            bbox_min=np.zeros(3),
            bbox_max=bs,
        )
        _log.debug("Wrote NDfield: %s", nd_path)

        # Find mse binary
        mse_bin = find_binary("mse", binary_dir=binary_dir)

        # Build args
        args = _build_mse_args(
            nd_path,
            periodic=periodic,
            persistence=persistence,
            smooth=smooth,
            extra_kwargs=kwargs,
        )
        cmd = [str(mse_bin)] + args
        commands.append(cmd)

        run_binary(mse_bin, args, cwd=wdir, extra_log=_log)

        # Try skelconv to ASCII for easier parsing
        try:
            skelconv_bin = find_binary("skelconv", binary_dir=binary_dir)
            out_ndskl = wdir / f"{stem}.NDskl"
            if out_ndskl.exists():
                skconv_args = [str(out_ndskl), "-to", "a.NDskl"]
                run_binary(skelconv_bin, skconv_args, cwd=wdir, extra_log=_log)
                commands.append([str(skelconv_bin)] + skconv_args)
        except Exception as exc:  # noqa: BLE001
            _log.debug("skelconv not available or failed: %s", exc)

        skeleton_files = _find_skeleton_files(wdir, stem)
        skeleton = _try_load_skeleton(skeleton_files)

    finally:
        if is_temp and not keep_intermediate:
            shutil.rmtree(wdir, ignore_errors=True)

    return DisperseRunResult(
        workdir=wdir,
        skeleton_files=skeleton_files,
        skeleton=skeleton,
        commands=commands,
        parameters={
            "mode": "points",
            "n_points": len(points_arr),
            "boxsize": bs.tolist(),
            "periodic": periodic,
            "persistence": persistence,
            "smooth": smooth,
        },
        elapsed_seconds=time.monotonic() - t0,
    )


def run_disperse_grid(
    grid: np.ndarray | str | os.PathLike,
    boxsize: float | tuple | list | np.ndarray,
    periodic: bool = True,
    *,
    workdir: str | os.PathLike | None = None,
    persistence: float | None = None,
    smooth: int | None = None,
    binary_dir: str | os.PathLike | None = None,
    keep_intermediate: bool = False,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> DisperseRunResult:
    """Run DisPerSE on a scalar density grid.

    Parameters
    ----------
    grid:
        Either a numpy array of shape ``(Nx, Ny, Nz)`` or a path to a
        ``.npy`` file containing such an array.
    boxsize:
        Physical side length(s) of the box.  Scalar or 3-tuple.
    periodic:
        Enable periodic boundary conditions.
    workdir:
        Directory for intermediate and output files.
    persistence:
        Persistence threshold in sigma units.
    smooth:
        Number of smoothing iterations.
    binary_dir:
        Directory containing DisPerSE binaries.
    keep_intermediate:
        Keep all intermediate files if *True*.
    logger:
        Optional Python logger.
    **kwargs:
        Additional flags for ``mse``.

    Returns
    -------
    DisperseRunResult
    """
    _log = logger or log
    t0 = time.monotonic()

    if isinstance(grid, (str, os.PathLike)):
        grid_arr = np.load(str(grid))
    else:
        grid_arr = grid  # type: ignore[assignment]
    grid_arr = _validate_grid(grid_arr)

    bs = _validate_boxsize(boxsize)

    wdir, is_temp = _make_workdir(workdir)
    commands: list[list[str]] = []
    stem = "grid"

    try:
        nd_path = write_ndfield(
            grid_arr,
            wdir / stem,
            bbox_min=np.zeros(3),
            bbox_max=bs,
        )
        _log.debug("Wrote NDfield: %s", nd_path)

        mse_bin = find_binary("mse", binary_dir=binary_dir)
        args = _build_mse_args(
            nd_path,
            periodic=periodic,
            persistence=persistence,
            smooth=smooth,
            extra_kwargs=kwargs,
        )
        cmd = [str(mse_bin)] + args
        commands.append(cmd)

        run_binary(mse_bin, args, cwd=wdir, extra_log=_log)

        try:
            skelconv_bin = find_binary("skelconv", binary_dir=binary_dir)
            out_ndskl = wdir / f"{stem}.NDskl"
            if out_ndskl.exists():
                skconv_args = [str(out_ndskl), "-to", "a.NDskl"]
                run_binary(skelconv_bin, skconv_args, cwd=wdir, extra_log=_log)
                commands.append([str(skelconv_bin)] + skconv_args)
        except Exception as exc:  # noqa: BLE001
            _log.debug("skelconv not available or failed: %s", exc)

        skeleton_files = _find_skeleton_files(wdir, stem)
        skeleton = _try_load_skeleton(skeleton_files)

    finally:
        if is_temp and not keep_intermediate:
            shutil.rmtree(wdir, ignore_errors=True)

    return DisperseRunResult(
        workdir=wdir,
        skeleton_files=skeleton_files,
        skeleton=skeleton,
        commands=commands,
        parameters={
            "mode": "grid",
            "grid_shape": list(grid_arr.shape),
            "boxsize": bs.tolist(),
            "periodic": periodic,
            "persistence": persistence,
            "smooth": smooth,
        },
        elapsed_seconds=time.monotonic() - t0,
    )


def load_skeleton(path: str | os.PathLike) -> Skeleton:
    """Load a DisPerSE skeleton from a file.

    Supported formats: ``.NDskl`` (binary), ``.a.NDskl`` (ASCII),
    ``.json``, ``.npz``, ``.hdf5``/``.h5``.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    Skeleton
    """
    return load_skeleton_file(Path(path))


def save_skeleton(
    skeleton: Skeleton,
    path: str | os.PathLike,
    fmt: str = "json",
) -> Path:
    """Save a :class:`~disperse_wrapper.skeleton.Skeleton` to a file.

    Parameters
    ----------
    skeleton : Skeleton
    path : str or Path
        Output file path.  The extension may be omitted when *fmt* matches.
    fmt : ``"json"``, ``"npz"``, or ``"hdf5"``

    Returns
    -------
    Path
        Path to the saved file.
    """
    return save_skeleton_file(skeleton, Path(path), fmt)
