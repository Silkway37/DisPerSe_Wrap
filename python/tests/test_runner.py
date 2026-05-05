"""Tests for binary discovery and subprocess runner."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from disperse_wrapper.exceptions import DisperseBinaryNotFoundError, DisperseRunError
from disperse_wrapper.runner import find_binary, run_binary

# ---------------------------------------------------------------------------
# find_binary
# ---------------------------------------------------------------------------


class TestFindBinary:
    def test_find_via_binary_dir(self, tmp_path: Path) -> None:
        exe = tmp_path / "mse"
        exe.write_text("#!/bin/sh\necho ok\n")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)

        found = find_binary("mse", binary_dir=tmp_path)
        assert found == exe.resolve()

    def test_find_via_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        exe = tmp_path / "skelconv"
        exe.write_text("#!/bin/sh\necho ok\n")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setenv("DISPERSE_BIN", str(tmp_path))
        found = find_binary("skelconv")
        assert found == exe.resolve()

    def test_find_via_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        exe = tmp_path / "fieldconv"
        exe.write_text("#!/bin/sh\necho ok\n")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
        monkeypatch.delenv("DISPERSE_BIN", raising=False)
        found = find_binary("fieldconv")
        assert found.name == "fieldconv"

    def test_not_found_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DISPERSE_BIN", raising=False)
        # Use a binary_dir that contains nothing
        with pytest.raises(DisperseBinaryNotFoundError) as exc_info:
            find_binary("mse", binary_dir=tmp_path)
        err = exc_info.value
        assert err.binary == "mse"
        assert str(tmp_path) in err.searched[0]
        assert "DISPERSE_BIN" in str(err)

    def test_binary_dir_takes_precedence_over_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        dir_b = tmp_path / "b"
        dir_b.mkdir()

        # binary in dir_a only
        exe_a = dir_a / "mse"
        exe_a.write_text("#!/bin/sh\n")
        exe_a.chmod(exe_a.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setenv("DISPERSE_BIN", str(dir_b))  # dir_b has NO binary

        found = find_binary("mse", binary_dir=dir_a)
        assert found == exe_a.resolve()

    def test_non_executable_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exe = tmp_path / "mse"
        exe.write_text("not a script")
        # deliberately NOT chmod +x
        monkeypatch.delenv("DISPERSE_BIN", raising=False)
        with pytest.raises(DisperseBinaryNotFoundError):
            find_binary("mse", binary_dir=tmp_path)


# ---------------------------------------------------------------------------
# run_binary
# ---------------------------------------------------------------------------


class TestRunBinary:
    def test_success(self, tmp_path: Path) -> None:
        exe = tmp_path / "hello"
        exe.write_text("#!/bin/sh\necho hello\n")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)

        result = run_binary(exe, [])
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_failure_raises(self, tmp_path: Path) -> None:
        exe = tmp_path / "fail"
        exe.write_text("#!/bin/sh\necho bad >&2\nexit 1\n")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)

        with pytest.raises(DisperseRunError) as exc_info:
            run_binary(exe, [])
        err = exc_info.value
        assert err.returncode == 1
        assert "bad" in err.stderr
        assert str(exe) in " ".join(err.command)

    def test_stderr_included_in_error(self, tmp_path: Path) -> None:
        exe = tmp_path / "nope"
        exe.write_text("#!/bin/sh\necho 'fatal error: something went wrong' >&2\nexit 2\n")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)

        with pytest.raises(DisperseRunError) as exc_info:
            run_binary(exe, [])
        assert "fatal error" in str(exc_info.value)
