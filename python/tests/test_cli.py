"""Tests for the CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from disperse_wrapper.cli import build_parser, main


class TestCLIParser:
    def test_run_points_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--points", "pts.npy",
                "--boxsize", "100.0",
                "--periodic",
                "--out", "/tmp/out",
                "--persistence", "3.0",
                "--smooth", "2",
                "--binary-dir", "/usr/local/bin",
                "--keep-intermediate",
            ]
        )
        assert args.points == "pts.npy"
        assert args.boxsize == 100.0
        assert args.periodic is True
        assert args.out == "/tmp/out"
        assert args.persistence == 3.0
        assert args.smooth == 2
        assert args.binary_dir == "/usr/local/bin"
        assert args.keep_intermediate is True

    def test_run_grid_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--grid", "g.npy", "--boxsize", "50.0"])
        assert args.grid == "g.npy"
        assert args.boxsize == 50.0

    def test_convert_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["convert", "--skel", "in.NDskl", "--to", "json", "--out", "out.json"]
        )
        assert args.skel == "in.NDskl"
        assert args.to == "json"
        assert args.out == "out.json"

    def test_info_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["info", "--skel", "skeleton.json"])
        assert args.skel == "skeleton.json"

    def test_missing_subcommand_fails(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestCLIInfo:
    def test_info_on_json(
        self,
        minimal_skeleton_text: str,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        from disperse_wrapper.api import save_skeleton
        from disperse_wrapper.io.skeleton_io import _parse_ascii_ndskl

        sk = _parse_ascii_ndskl(minimal_skeleton_text)
        json_file = tmp_path / "skel.json"
        save_skeleton(sk, json_file, fmt="json")

        ret = main(["info", "--skel", str(json_file)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Nodes" in out
        assert "Arcs" in out


class TestCLIConvert:
    def test_convert_json_to_npz(
        self,
        minimal_skeleton_text: str,
        tmp_path: Path,
    ) -> None:
        from disperse_wrapper.api import save_skeleton
        from disperse_wrapper.io.skeleton_io import _parse_ascii_ndskl

        sk = _parse_ascii_ndskl(minimal_skeleton_text)
        json_file = tmp_path / "skel.json"
        save_skeleton(sk, json_file, fmt="json")

        npz_file = tmp_path / "skel.npz"
        ret = main(["convert", "--skel", str(json_file), "--to", "npz", "--out", str(npz_file)])
        assert ret == 0
        assert npz_file.exists()


class TestCLIRun:
    def test_run_points(
        self,
        fake_mse_bin: Path,
        sample_points: np.ndarray,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        npy_path = tmp_path / "pts.npy"
        np.save(npy_path, sample_points)
        out_dir = tmp_path / "out"

        ret = main([
            "run",
            "--points", str(npy_path),
            "--boxsize", "100.0",
            "--periodic",
            "--out", str(out_dir),
            "--binary-dir", str(fake_mse_bin),
            "--keep-intermediate",
        ])
        assert ret == 0
        out = capsys.readouterr().out
        assert "complete" in out.lower()
