"""Custom exceptions for the DisPerSe wrapper."""

from __future__ import annotations


class DisperseBinaryNotFoundError(FileNotFoundError):
    """Raised when a required DisPerSE binary cannot be located.

    Attributes
    ----------
    binary : str
        Name of the binary that could not be found.
    searched : list[str]
        Directories / env-vars that were searched.
    """

    def __init__(self, binary: str, searched: list[str] | None = None) -> None:
        self.binary = binary
        self.searched = searched or []
        search_info = (
            "\n  Searched in: " + ", ".join(self.searched) if self.searched else ""
        )
        super().__init__(
            f"DisPerSE binary '{binary}' not found.{search_info}\n"
            "Fix options:\n"
            "  1. Set the DISPERSE_BIN environment variable to the directory "
            "containing the DisPerSE binaries.\n"
            "  2. Add the directory to your PATH.\n"
            "  3. Pass binary_dir=<path> to the run function.\n"
            "  4. Install DisPerSE: https://www2.iap.fr/users/sousbie/web/html/indexd41d.html"
        )


class DisperseRunError(RuntimeError):
    """Raised when a DisPerSE binary exits with a non-zero status.

    Attributes
    ----------
    command : list[str]
        The full command that was executed.
    returncode : int
        The exit code of the process.
    stdout : str
        Standard output captured from the process.
    stderr : str
        Standard error captured from the process.
    """

    _MAX_OUTPUT_LEN = 4000

    def __init__(
        self,
        command: list[str],
        returncode: int,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

        cmd_str = " ".join(str(c) for c in command)
        stdout_tail = self._tail(stdout)
        stderr_tail = self._tail(stderr)

        super().__init__(
            f"DisPerSE command failed (exit code {returncode}):\n"
            f"  Command: {cmd_str}\n"
            + (f"  stdout:\n{stdout_tail}\n" if stdout_tail else "")
            + (f"  stderr:\n{stderr_tail}\n" if stderr_tail else "")
        )

    def _tail(self, text: str) -> str:
        if not text:
            return ""
        if len(text) > self._MAX_OUTPUT_LEN:
            return "...(truncated)...\n" + text[-self._MAX_OUTPUT_LEN :]
        return text


class DisperseInputError(ValueError):
    """Raised when the caller supplies invalid input (bad shape, negative boxsize, …)."""
