"""Shared pytest fixtures."""

from __future__ import annotations

import stat
import textwrap
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def fake_mse_bin(tmp_path: Path) -> Path:
    """Create a fake 'mse' executable that creates a minimal .NDskl file."""
    bin_dir = tmp_path / "fake_bins"
    bin_dir.mkdir()

    # Minimal ASCII NDskl content that our parser can read
    ndskl_template = textwrap.dedent("""\
        NDSKEL
        NDIMS 3
        NVERTEX 4
        NFILAMENT 2
        BBOX 0.0 0.0 0.0 100.0 100.0 100.0
        CRITICAL POINTS
        10.0 20.0 30.0 0.5 1 0
        50.0 50.0 50.0 1.2 0 3
        20.0 40.0 60.0 0.8 3 1
        80.0 80.0 80.0 0.3 2 2
        FILAMENTS
        0 1 3
        10.0 20.0 30.0
        25.0 35.0 40.0
        50.0 50.0 50.0
        2 3 2
        20.0 40.0 60.0
        80.0 80.0 80.0
    """)

    script = bin_dir / "mse"
    script.write_text(
        "#!/bin/sh\n"
        # find the first argument that looks like a file (the input NDfield)
        "INPUT=$1\n"
        # derive output stem
        "STEM=${INPUT%.NDfield}\n"
        f"cat > ${{STEM}}.NDskl << 'HEREDOC'\n"
        f"{ndskl_template}\n"
        "HEREDOC\n"
        "exit 0\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    # Also create a fake skelconv that copies .NDskl to .a.NDskl
    skelconv = bin_dir / "skelconv"
    skelconv.write_text(
        "#!/bin/sh\n"
        "INPUT=$1\n"
        'if [ "$3" = "a.NDskl" ]; then\n'
        '  cp "$INPUT" "${INPUT%.NDskl}.a.NDskl"\n'
        "fi\n"
        "exit 0\n"
    )
    skelconv.chmod(skelconv.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return bin_dir


@pytest.fixture()
def minimal_skeleton_text() -> str:
    return textwrap.dedent("""\
        NDSKEL
        NDIMS 3
        NVERTEX 4
        NFILAMENT 2
        BBOX 0.0 0.0 0.0 100.0 100.0 100.0
        CRITICAL POINTS
        10.0 20.0 30.0 0.5 1 0
        50.0 50.0 50.0 1.2 0 3
        20.0 40.0 60.0 0.8 3 1
        80.0 80.0 80.0 0.3 2 2
        FILAMENTS
        0 1 3
        10.0 20.0 30.0
        25.0 35.0 40.0
        50.0 50.0 50.0
        2 3 2
        20.0 40.0 60.0
        80.0 80.0 80.0
    """)


@pytest.fixture()
def sample_points() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(0, 100, size=(200, 3))


@pytest.fixture()
def sample_grid() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, size=(16, 16, 16)).astype(np.float32)
