"""disperse_wrapper – Python wrapper for DisPerSE topology analysis tools.

Quick start
-----------
>>> from disperse_wrapper import run_disperse_points, load_skeleton, save_skeleton
>>> result = run_disperse_points(
...     points,          # np.ndarray shape (N, 3)
...     boxsize=100.0,
...     periodic=True,
...     persistence=3.0,
...     workdir="./my_run",
... )
>>> print(result.skeleton.summary())
>>> save_skeleton(result.skeleton, "skeleton.json")
"""

from .api import (
    DisperseRunResult,
    load_skeleton,
    run_disperse_grid,
    run_disperse_points,
    save_skeleton,
)
from .exceptions import (
    DisperseBinaryNotFoundError,
    DisperseInputError,
    DisperseRunError,
)
from .skeleton import Arc, CriticalPoint, Skeleton

__all__ = [
    # API
    "run_disperse_points",
    "run_disperse_grid",
    "load_skeleton",
    "save_skeleton",
    "DisperseRunResult",
    # Skeleton data model
    "Skeleton",
    "CriticalPoint",
    "Arc",
    # Exceptions
    "DisperseBinaryNotFoundError",
    "DisperseRunError",
    "DisperseInputError",
]

__version__ = "0.1.0"
