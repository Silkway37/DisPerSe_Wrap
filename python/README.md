# DisPerSE Wrapper

A modern, installable Python package that wraps the
[DisPerSE](https://www2.iap.fr/users/sousbie/web/html/indexd41d.html)
command-line tools for cosmic-web topology analysis.

## Features

- Run DisPerSE on raw **point clouds** (N×3 galaxy/sub-halo positions).
- Run DisPerSE on **gridded scalar fields** (density grids).
- Full support for **periodic boundary conditions**.
- Clean Python object model for the skeleton (nodes, arcs, metadata).
- Save/load skeletons as **JSON, NumPy NPZ, or HDF5**.
- Optional export to **NetworkX** graphs.
- **PySide6 GUI** (optional) for interactive use.
- `disperse-wrapper` **CLI** for scripted workflows.

---

## Quick start

### 1 – Install the Python package

```bash
# Core package (no GUI, no graph, no HDF5)
pip install -e python

# With all extras
pip install -e "python[gui,graph,io,dev]"
```

### 2 – Point the wrapper at the DisPerSE binaries

DisPerSE ships its own compiled binaries (`mse`, `skelconv`, …).
You must compile or download them separately.

```bash
# Option A: environment variable (recommended)
export DISPERSE_BIN=/path/to/disperse/bin

# Option B: add to PATH
export PATH=/path/to/disperse/bin:$PATH
```

### 3 – Run from Python

```python
import numpy as np
from disperse_wrapper import run_disperse_points, load_skeleton, save_skeleton

# Load or create your point cloud
points = np.load("subhalos.npy")   # shape (N, 3)

result = run_disperse_points(
    points,
    boxsize=100.0,        # Mpc/h
    periodic=True,        # periodic boundary conditions
    persistence=3.0,      # sigma threshold
    workdir="./my_run",
)

print(result.skeleton.summary())

# Save the skeleton
save_skeleton(result.skeleton, "skeleton.json")
```

### 4 – Run from the CLI

```bash
# Point cloud
disperse-wrapper run \
    --points subhalos.npy \
    --boxsize 100.0 \
    --periodic \
    --persistence 3.0 \
    --out ./output

# Density grid
disperse-wrapper run \
    --grid density.npy \
    --boxsize 100.0 \
    --out ./output

# Convert skeleton format
disperse-wrapper convert --skel output/points.NDskl --to json --out skeleton.json

# Print statistics
disperse-wrapper info --skel skeleton.json
```

---

## Installing DisPerSE binaries

### Option A – Build from source

```bash
git clone https://github.com/thierry-sousbie/DisPerSE.git
cd DisPerSE
mkdir build && cd build
cmake ..
make -j4
# binaries are in build/bin/
export DISPERSE_BIN=$(pwd)/bin
```

### Option B – Container (Docker)

```bash
# Pull or build the container
docker build -t disperse .

# Run through the wrapper (mount your data directory)
docker run --rm -v $PWD:/data disperse \
    mse /data/points.NDfield -periodic -nsig 3
```

### Option C – Apptainer / Singularity

```bash
apptainer pull disperse.sif docker://your_registry/disperse:latest
apptainer exec disperse.sif mse points.NDfield -periodic -nsig 3
```

---

## Conda environment

```bash
conda env create -f python/environment.yml
conda activate disperse_wrapper
pip install -e "python[graph,io,dev]"
```

---

## Running the GUI

```bash
pip install -e "python[gui]"
disperse-wrapper-gui
# or
python -m disperse_wrapper_gui
```

---

## Measuring WHIM profiles around filaments

After running DisPerSE you can measure Warm-Hot Intergalactic Medium (WHIM)
profiles around filaments:

```python
from disperse_wrapper import load_skeleton

sk = load_skeleton("skeleton.json")
arcs = sk.arcs

# For each arc, extract the polyline
for arc in arcs[:5]:
    print(f"Arc {arc.id}: {len(arc.points)} samples, length={arc.length:.2f} Mpc/h")
    # arc.points is (n_samples, 3) – use these coordinates to
    # query your hydro simulation or observational data
```

---

## API Reference

### `run_disperse_points(points, boxsize, periodic=True, *, ...)`

Run DisPerSE on a point cloud.

| Argument | Type | Description |
|---|---|---|
| `points` | `np.ndarray (N,3)` or `.npy` path | Particle/galaxy positions |
| `boxsize` | float or 3-tuple | Physical box side length |
| `periodic` | bool | Enable periodic BCs (`-periodic` flag) |
| `persistence` | float or None | Persistence threshold in σ |
| `smooth` | int or None | Smoothing iterations |
| `workdir` | path or None | Output directory (temp if None) |
| `binary_dir` | path or None | Directory containing DisPerSE binaries |
| `keep_intermediate` | bool | Keep all working files |

Returns `DisperseRunResult`.

### `run_disperse_grid(grid, boxsize, periodic=True, *, ...)`

Same as above but for a 3-D density grid `(Nx, Ny, Nz)`.

### `load_skeleton(path)` / `save_skeleton(skeleton, path, fmt)`

Load from / save to `.json`, `.npz`, `.hdf5`, `.NDskl`, or `.a.NDskl`.

### `Skeleton`

| Property | Description |
|---|---|
| `n_nodes` | Number of critical points |
| `n_arcs` | Number of filaments |
| `total_length` | Sum of all arc lengths |
| `minima / maxima / saddles_1 / saddles_2` | Filtered node lists |
| `to_numpy()` | Dict of NumPy arrays |
| `to_networkx()` | NetworkX graph (requires `networkx`) |
| `summary()` | Human-readable statistics string |

---

## Development

```bash
cd python
pip install -e ".[dev]"
pytest
ruff check .
```
