"""PySide6 GUI for DisPerSE Wrapper.

Provides a minimal but functional interface to run DisPerSE through the
Python wrapper, display progress logs, and show a summary of the resulting
skeleton.
"""

from __future__ import annotations

import logging
import sys
import traceback

import numpy as np

# Maximum characters shown in the error dialog
_ERROR_DIALOG_MAX_LEN = 500

try:
    from PySide6.QtCore import QObject, QThread, Signal, Slot
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QSpinBox,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as _e:
    raise ImportError(
        "PySide6 is required for the GUI.  "
        "Install it with: pip install disperse-wrapper[gui]"
    ) from _e


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


class _LogHandler(logging.Handler, QObject):
    """A logging.Handler that emits a Qt signal for each log record."""

    log_emitted = Signal(str)

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.log_emitted.emit(msg)
        except Exception:  # noqa: BLE001
            pass


class RunWorker(QObject):
    """Worker that runs DisPerSE in a background QThread."""

    finished = Signal(object)   # DisperseRunResult or None
    error = Signal(str)         # error message
    log_line = Signal(str)      # log output lines

    def __init__(
        self,
        mode: str,
        input_path: str,
        boxsize: float,
        periodic: bool,
        persistence: float | None,
        smooth: int | None,
        out_dir: str,
        binary_dir: str,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._input_path = input_path
        self._boxsize = boxsize
        self._periodic = periodic
        self._persistence = persistence
        self._smooth = smooth
        self._out_dir = out_dir or None
        self._binary_dir = binary_dir or None

    @Slot()
    def run(self) -> None:
        from disperse_wrapper import run_disperse_grid, run_disperse_points

        # Set up a logger that feeds into the Qt signal
        _log = logging.getLogger("disperse_wrapper.gui_run")
        _log.setLevel(logging.DEBUG)
        handler = _LogHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        handler.log_emitted.connect(self.log_line)
        _log.addHandler(handler)

        try:
            kwargs: dict = {"logger": _log, "keep_intermediate": True}
            if self._binary_dir:
                kwargs["binary_dir"] = self._binary_dir
            if self._out_dir:
                kwargs["workdir"] = self._out_dir

            if self._mode == "points":
                arr = np.load(self._input_path)
                result = run_disperse_points(
                    arr,
                    self._boxsize,
                    periodic=self._periodic,
                    persistence=self._persistence,
                    smooth=self._smooth if self._smooth > 0 else None,
                    **kwargs,
                )
            else:
                result = run_disperse_grid(
                    self._input_path,
                    self._boxsize,
                    periodic=self._periodic,
                    persistence=self._persistence,
                    smooth=self._smooth if self._smooth > 0 else None,
                    **kwargs,
                )

            self.finished.emit(result)
        except Exception:  # noqa: BLE001
            self.error.emit(traceback.format_exc())
        finally:
            _log.removeHandler(handler)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DisPerSE Wrapper")
        self.setMinimumWidth(700)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)

        # ---- Input type ----
        input_type_box = QGroupBox("Input type")
        it_layout = QHBoxLayout(input_type_box)
        self._rb_points = QRadioButton("Point cloud (N×3 .npy)")
        self._rb_grid = QRadioButton("Density grid (Nx×Ny×Nz .npy)")
        self._rb_points.setChecked(True)
        it_layout.addWidget(self._rb_points)
        it_layout.addWidget(self._rb_grid)
        main_layout.addWidget(input_type_box)

        # ---- File selection ----
        file_box = QGroupBox("Input file")
        file_layout = QHBoxLayout(file_box)
        self._input_path_edit = QLineEdit()
        self._input_path_edit.setPlaceholderText("Select a .npy file …")
        browse_btn = QPushButton("Browse …")
        browse_btn.clicked.connect(self._browse_input)
        file_layout.addWidget(self._input_path_edit)
        file_layout.addWidget(browse_btn)
        main_layout.addWidget(file_box)

        # ---- Parameters ----
        params_box = QGroupBox("Parameters")
        form = QFormLayout(params_box)

        self._boxsize_spin = QDoubleSpinBox()
        self._boxsize_spin.setRange(1e-6, 1e12)
        self._boxsize_spin.setValue(100.0)
        self._boxsize_spin.setSuffix("  (physical units)")
        form.addRow("Box size:", self._boxsize_spin)

        self._periodic_cb = QCheckBox("Periodic boundary conditions")
        self._periodic_cb.setChecked(True)
        form.addRow("", self._periodic_cb)

        self._persistence_spin = QDoubleSpinBox()
        self._persistence_spin.setRange(0.0, 1e6)
        self._persistence_spin.setSpecialValueText("(default)")
        self._persistence_spin.setValue(0.0)
        form.addRow("Persistence threshold (σ):", self._persistence_spin)

        self._smooth_spin = QSpinBox()
        self._smooth_spin.setRange(0, 100)
        self._smooth_spin.setValue(0)
        self._smooth_spin.setSpecialValueText("(none)")
        form.addRow("Smoothing iterations:", self._smooth_spin)

        # Output dir
        out_layout = QHBoxLayout()
        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setPlaceholderText("(temporary directory)")
        out_browse = QPushButton("Browse …")
        out_browse.clicked.connect(self._browse_out)
        out_layout.addWidget(self._out_dir_edit)
        out_layout.addWidget(out_browse)
        form.addRow("Output directory:", out_layout)

        # Binary dir
        bin_layout = QHBoxLayout()
        self._binary_dir_edit = QLineEdit()
        self._binary_dir_edit.setPlaceholderText("(auto-detect from PATH / DISPERSE_BIN)")
        bin_browse = QPushButton("Browse …")
        bin_browse.clicked.connect(self._browse_bin)
        bin_layout.addWidget(self._binary_dir_edit)
        bin_layout.addWidget(bin_browse)
        form.addRow("DisPerSE binary dir:", bin_layout)

        main_layout.addWidget(params_box)

        # ---- Run button ----
        self._run_btn = QPushButton("▶  Run DisPerSE")
        self._run_btn.setFixedHeight(36)
        self._run_btn.clicked.connect(self._on_run)
        main_layout.addWidget(self._run_btn)

        # ---- Log window ----
        log_box = QGroupBox("Progress / Log")
        log_layout = QVBoxLayout(log_box)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("Courier", 9))
        self._log_text.setMinimumHeight(160)
        log_layout.addWidget(self._log_text)
        main_layout.addWidget(log_box)

        # ---- Summary ----
        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        main_layout.addWidget(self._summary_label)

        self._thread: QThread | None = None
        self._worker: RunWorker | None = None

    # ------------------------------------------------------------------

    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select input .npy file", "", "NumPy (*.npy);;All Files (*)")
        if path:
            self._input_path_edit.setText(path)

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self._out_dir_edit.setText(path)

    def _browse_bin(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select DisPerSE binary directory")
        if path:
            self._binary_dir_edit.setText(path)

    def _append_log(self, text: str) -> None:
        self._log_text.append(text)
        # auto-scroll
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_run(self) -> None:
        input_path = self._input_path_edit.text().strip()
        if not input_path:
            QMessageBox.warning(self, "Missing input", "Please select an input .npy file.")
            return

        mode = "points" if self._rb_points.isChecked() else "grid"
        boxsize = self._boxsize_spin.value()
        periodic = self._periodic_cb.isChecked()
        persistence = self._persistence_spin.value() or None
        smooth = self._smooth_spin.value()
        out_dir = self._out_dir_edit.text().strip()
        binary_dir = self._binary_dir_edit.text().strip()

        self._log_text.clear()
        self._summary_label.setText("")
        self._run_btn.setEnabled(False)

        self._worker = RunWorker(
            mode=mode,
            input_path=input_path,
            boxsize=boxsize,
            periodic=periodic,
            persistence=persistence,
            smooth=smooth,
            out_dir=out_dir,
            binary_dir=binary_dir,
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.log_line.connect(self._append_log)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self._run_btn.setEnabled(True))
        self._thread.start()

    @Slot(object)
    def _on_finished(self, result) -> None:
        self._append_log(f"\n✓ Run completed in {result.elapsed_seconds:.2f}s")
        self._append_log(f"  Work dir: {result.workdir}")
        if result.skeleton:
            self._summary_label.setText(result.skeleton.summary().replace("\n", "<br>"))
        else:
            self._summary_label.setText("No skeleton could be loaded automatically.")

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._append_log(f"\n✗ ERROR:\n{msg}")
        QMessageBox.critical(self, "DisPerSE error", msg[:_ERROR_DIALOG_MAX_LEN])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
