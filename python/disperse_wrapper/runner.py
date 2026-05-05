"""Binary discovery and subprocess runner for DisPerSE tools."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .exceptions import DisperseBinaryNotFoundError, DisperseRunError

logger = logging.getLogger(__name__)

# Canonical set of DisPerSE binaries used by this wrapper
DISPERSE_BINARIES = ("mse", "skelconv", "fieldconv", "addss")

# Maximum number of characters logged from stdout/stderr per binary invocation
_MAX_LOG_OUTPUT_LEN = 2000


def find_binary(
    name: str,
    binary_dir: str | os.PathLike | None = None,
) -> Path:
    """Return the absolute path to a DisPerSE binary.

    Search order
    ------------
    1. *binary_dir* argument (if given).
    2. ``DISPERSE_BIN`` environment variable.
    3. ``PATH``.

    Parameters
    ----------
    name:
        Binary name, e.g. ``"mse"`` or ``"skelconv"``.
    binary_dir:
        Optional explicit directory containing DisPerSE binaries.

    Returns
    -------
    Path
        Absolute path to the binary.

    Raises
    ------
    DisperseBinaryNotFoundError
        If the binary cannot be located.
    """
    searched: list[str] = []

    # 1. Explicit binary_dir
    if binary_dir is not None:
        candidate = Path(binary_dir) / name
        searched.append(str(Path(binary_dir)))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    # 2. DISPERSE_BIN env var
    env_dir = os.environ.get("DISPERSE_BIN")
    if env_dir:
        candidate = Path(env_dir) / name
        searched.append(f"$DISPERSE_BIN ({env_dir})")
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    # 3. PATH
    which = shutil.which(name)
    searched.append("$PATH")
    if which:
        return Path(which).resolve()

    raise DisperseBinaryNotFoundError(name, searched)


def run_binary(
    binary_path: Path | str,
    args: Sequence[str | os.PathLike],
    *,
    cwd: Path | str | None = None,
    extra_log: logging.Logger | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute a DisPerSE binary and return the completed-process object.

    Parameters
    ----------
    binary_path:
        Full path to (or name of) the binary.
    args:
        Command-line arguments to pass after the binary name.
    cwd:
        Working directory for the subprocess.
    extra_log:
        Optional logger to receive stdout/stderr at DEBUG level.

    Returns
    -------
    subprocess.CompletedProcess[str]

    Raises
    ------
    DisperseRunError
        If the process exits with a non-zero return code.
    """
    cmd: list[str] = [str(binary_path)] + [str(a) for a in args]
    log = extra_log or logger
    log.debug("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        raise DisperseRunError(cmd, -1, "", str(exc)) from exc

    if result.stdout:
        log.debug("stdout: %s", result.stdout[-_MAX_LOG_OUTPUT_LEN:])
    if result.stderr:
        log.debug("stderr: %s", result.stderr[-_MAX_LOG_OUTPUT_LEN:])

    if result.returncode != 0:
        raise DisperseRunError(cmd, result.returncode, result.stdout, result.stderr)

    return result


def check_binaries(
    *names: str,
    binary_dir: str | os.PathLike | None = None,
) -> dict[str, Path]:
    """Check that all named binaries are findable.

    Returns
    -------
    dict[str, Path]
        Mapping of binary name → absolute path.

    Raises
    ------
    DisperseBinaryNotFoundError
        On the first binary that cannot be found.
    """
    return {name: find_binary(name, binary_dir=binary_dir) for name in names}
