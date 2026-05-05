"""Tests for the high-level API (parameter validation + integration with fake binaries)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from disperse_wrapper.api import (
    DisperseRunResult,
    run_disperse_grid,
    run_disperse_points,
)
from disperse_wrapper.exceptions import DisperseBinaryNotFoundError, DisperseInputError

# ---------------------------------------------------------------------------
# Validation tests (no binaries required)
# ---------------------------------------------------------------------------


class TestValidation:
    """These tests exercise input validation before any binary is invoked."""

    def test_points_wrong_shape_1d(self, tmp_path: Path) -> None:
        bad = np.ones(30)
        with pytest.raises(DisperseInputError, match="shape"):
            run_disperse_points(bad, boxsize=100.0, binary_dir=tmp_path)

    def test_points_wrong_ndim(self, tmp_path: Path) -> None:
        bad = np.ones((10, 2))
        with pytest.raises(DisperseInputError, match="shape"):
            run_disperse_points(bad, boxsize=100.0, binary_dir=tmp_path)

    def test_points_too_few(self, tmp_path: Path) -> None:
        bad = np.ones((3, 3))
        with pytest.raises(DisperseInputError, match="at least 4"):
            run_disperse_points(bad, boxsize=100.0, binary_dir=tmp_path)

    def test_boxsize_negative(self, tmp_path: Path) -> None:
        pts = np.random.rand(10, 3)
        with pytest.raises(DisperseInputError, match="positive"):
            run_disperse_points(pts, boxsize=-1.0, binary_dir=tmp_path)

    def test_boxsize_wrong_length(self, tmp_path: Path) -> None:
        pts = np.random.rand(10, 3)
        with pytest.raises(DisperseInputError):
            run_disperse_points(pts, boxsize=(1.0, 2.0), binary_dir=tmp_path)

    def test_grid_wrong_shape(self, tmp_path: Path) -> None:
        bad = np.ones((10, 10))
        with pytest.raises(DisperseInputError, match="shape"):
            run_disperse_grid(bad, boxsize=100.0, binary_dir=tmp_path)

    def test_missing_binary(self, tmp_path: Path) -> None:
        pts = np.random.rand(50, 3)
        empty_bin_dir = tmp_path / "no_bins"
        empty_bin_dir.mkdir()
        with pytest.raises(DisperseBinaryNotFoundError):
            run_disperse_points(pts, boxsize=100.0, binary_dir=empty_bin_dir)

    def test_boxsize_tuple(self, tmp_path: Path) -> None:
        """3-tuple boxsize should not raise."""
        pts = np.random.rand(50, 3)
        empty_bin_dir = tmp_path / "no_bins"
        empty_bin_dir.mkdir()
        # should raise DisperseBinaryNotFoundError, NOT DisperseInputError
        with pytest.raises(DisperseBinaryNotFoundError):
            run_disperse_points(pts, boxsize=(100.0, 100.0, 100.0), binary_dir=empty_bin_dir)


# ---------------------------------------------------------------------------
# Integration tests (use fake mse binary)
# ---------------------------------------------------------------------------


class TestRunDispersePoints:
    def test_basic_run(
        self,
        fake_mse_bin: Path,
        sample_points: np.ndarray,
        tmp_path: Path,
    ) -> None:
        result = run_disperse_points(
            sample_points,
            boxsize=100.0,
            periodic=True,
            binary_dir=fake_mse_bin,
            workdir=tmp_path / "run1",
            keep_intermediate=True,
        )
        assert isinstance(result, DisperseRunResult)
        assert result.elapsed_seconds >= 0
        assert len(result.skeleton_files) > 0
        # skeleton should have been parsed
        assert result.skeleton is not None
        sk = result.skeleton
        assert sk.n_nodes == 4
        assert sk.n_arcs == 2

    def test_persistence_flag_passed(
        self,
        fake_mse_bin: Path,
        sample_points: np.ndarray,
        tmp_path: Path,
    ) -> None:
        result = run_disperse_points(
            sample_points,
            boxsize=100.0,
            persistence=3.0,
            binary_dir=fake_mse_bin,
            workdir=tmp_path / "run2",
            keep_intermediate=True,
        )
        # Check that -nsig 3.0 was in the command
        all_cmds_flat = " ".join(" ".join(str(x) for x in cmd) for cmd in result.commands)
        assert "-nsig" in all_cmds_flat
        assert "3.0" in all_cmds_flat

    def test_periodic_flag_passed(
        self,
        fake_mse_bin: Path,
        sample_points: np.ndarray,
        tmp_path: Path,
    ) -> None:
        result = run_disperse_points(
            sample_points,
            boxsize=100.0,
            periodic=True,
            binary_dir=fake_mse_bin,
            workdir=tmp_path / "run3",
            keep_intermediate=True,
        )
        all_cmds_flat = " ".join(" ".join(str(x) for x in cmd) for cmd in result.commands)
        assert "-periodic" in all_cmds_flat

    def test_non_periodic_no_flag(
        self,
        fake_mse_bin: Path,
        sample_points: np.ndarray,
        tmp_path: Path,
    ) -> None:
        result = run_disperse_points(
            sample_points,
            boxsize=100.0,
            periodic=False,
            binary_dir=fake_mse_bin,
            workdir=tmp_path / "run4",
            keep_intermediate=True,
        )
        all_cmds_flat = " ".join(" ".join(str(x) for x in cmd) for cmd in result.commands)
        assert "-periodic" not in all_cmds_flat

    def test_load_from_npy_path(
        self,
        fake_mse_bin: Path,
        sample_points: np.ndarray,
        tmp_path: Path,
    ) -> None:
        npy_path = tmp_path / "pts.npy"
        np.save(npy_path, sample_points)

        result = run_disperse_points(
            npy_path,
            boxsize=100.0,
            binary_dir=fake_mse_bin,
            workdir=tmp_path / "run5",
            keep_intermediate=True,
        )
        assert result.skeleton is not None


class TestRunDisperseGrid:
    def test_basic_run(
        self,
        fake_mse_bin: Path,
        sample_grid: np.ndarray,
        tmp_path: Path,
    ) -> None:
        result = run_disperse_grid(
            sample_grid,
            boxsize=100.0,
            periodic=True,
            binary_dir=fake_mse_bin,
            workdir=tmp_path / "grid_run",
            keep_intermediate=True,
        )
        assert isinstance(result, DisperseRunResult)
        assert result.skeleton is not None
        assert result.parameters["mode"] == "grid"
        assert result.parameters["grid_shape"] == [16, 16, 16]
