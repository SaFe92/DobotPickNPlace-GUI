import os
import sys
import threading
import traceback
from datetime import datetime

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from config_manager import CONFIG_PATH, get_float, get_int, get_str
from dobot_pick import DobotE6


COLOR_WARNING = "#e6a100"
COLOR_ERROR = "#e05555"

def open_session_log():
    base_dir = os.path.dirname(os.path.abspath(CONFIG_PATH))
    out_dir = os.path.normpath(os.path.join(base_dir, get_str("Camera", "OUTPUT_DIR", "./output")))
    try:
        os.makedirs(out_dir, exist_ok=True)
        name = datetime.now().strftime("session_%Y%m%d_%H%M%S.log")
        return open(os.path.join(out_dir, name), "w", encoding="utf-8")
    except OSError as exc:
        print(f"WARNING: Could not create session log file: {exc}", file=sys.__stderr__)
        return None


class LogStream(QObject):
    line_written = Signal(str, str)  # (formatted line, level)

    def __init__(self, original_stream, level, lock, log_file):
        super().__init__()
        self._original = original_stream
        self._level = level          # "info" for stdout, "error" for stderr
        self._lock = lock            # shared by both streams, so lines don't mix
        self._log_file = log_file
        self._buffer = ""

    def write(self, text):
        if not text:
            return
        self._original.write(text)
        with self._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    self._emit_line(line)

    def _emit_line(self, line):
        stamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{stamp}] {line}"

        level = self._level
        if level == "info" and line.upper().startswith("WARN"):
            level = "warning"

        if self._log_file is not None:
            try:
                self._log_file.write(formatted + "\n")
                self._log_file.flush()
            except OSError:
                pass  # a full disk shouldn't stop the GUI

        self.line_written.emit(formatted, level)

    def flush(self):
        self._original.flush()


class Task(QRunnable):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.fn()
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dobot Pick-and-Place")
        self.resize(900, 600)

        self.robot = None

        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self.log_file = open_session_log()
        lock = threading.Lock()

        self.out_stream = LogStream(self._orig_stdout, "info", lock, self.log_file)
        self.err_stream = LogStream(self._orig_stderr, "error", lock, self.log_file)
        self.out_stream.line_written.connect(self.append_log)
        self.err_stream.line_written.connect(self.append_log)
        sys.stdout = self.out_stream
        sys.stderr = self.err_stream

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_robot_group())
        layout.addWidget(self.log, stretch=1)
        self.setCentralWidget(central)

        if self.log_file is not None:
            print(f"Session log: {self.log_file.name}")
        print("Ready. Enter the robot's IP and port, then press Connect.")

    def append_log(self, line, level):
        """Add one coloured line at the end of the log and scroll to it."""
        fmt = QTextCharFormat()
        if level == "error":
            fmt.setForeground(QColor(COLOR_ERROR))
        elif level == "warning":
            fmt.setForeground(QColor(COLOR_WARNING))

        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(line + "\n", fmt)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def closeEvent(self, event):
        print("Session ended.")
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        if self.log_file is not None:
            self.log_file.close()
        super().closeEvent(event)

    def _build_connection_group(self):
        box = QGroupBox("Connection")
        row = QHBoxLayout(box)

        self.ip_edit = QLineEdit(get_str("Robot Connection", "ROBOT_IP", "192.168.5.1"))
        self.port_edit = QLineEdit(str(get_int("Robot Connection", "ROBOT_PORT", 29999)))
        self.port_edit.setFixedWidth(70)

        connect_btn = QPushButton("Connect")
        connect_btn.clicked.connect(self.connect_robot)
        disconnect_btn = QPushButton("Disconnect")
        disconnect_btn.clicked.connect(self.disconnect_robot)

        row.addWidget(QLabel("IP:"))
        row.addWidget(self.ip_edit)
        row.addWidget(QLabel("Port:"))
        row.addWidget(self.port_edit)
        row.addWidget(connect_btn)
        row.addWidget(disconnect_btn)
        return box

    def _build_robot_group(self):
        box = QGroupBox("Robot")
        row = QHBoxLayout(box)

        enable_btn = QPushButton("Enable")
        enable_btn.clicked.connect(lambda: self.run_robot(lambda: self.robot.enable()))

        clear_btn = QPushButton("Clear Error")
        clear_btn.clicked.connect(lambda: self.run_robot(lambda: self.robot.clear_error()))

        home_btn = QPushButton("Home")
        home_btn.clicked.connect(self.go_home)

        estop_btn = QPushButton("EMERGENCY STOP")
        estop_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        estop_btn.clicked.connect(self.emergency_stop)

        for btn in (enable_btn, clear_btn, home_btn):
            row.addWidget(btn)
        row.addStretch()
        row.addWidget(estop_btn)
        return box

    def connect_robot(self):
        try:
            port = int(self.port_edit.text())
        except ValueError:
            print("ERROR: Port must be a number.", file=sys.stderr)
            return
        self.robot = DobotE6(self.ip_edit.text().strip(), port)
        self.run_robot(self.robot.connect)

    def disconnect_robot(self):
        if self.robot is not None:
            self.run_robot(self.robot.disconnect)

    def go_home(self):
        pose = (
            get_float("Robot Position", "HOME_X", 0.0),
            get_float("Robot Position", "HOME_Y", 0.0),
            get_float("Robot Position", "HOME_Z", 0.0),
            get_float("Robot Position", "HOME_RX", 0.0),
            get_float("Robot Position", "HOME_RY", 0.0),
            get_float("Robot Position", "HOME_RZ", 0.0),
        )
        self.run_robot(lambda: self.robot.move_to(*pose))

    def emergency_stop(self):
        if self.robot is None:
            print("EMERGENCY STOP: not connected.", file=sys.stderr)
            return
        print("EMERGENCY STOP")
        threading.Thread(target=self.robot.disable, daemon=True).start()

    def run_robot(self, fn):
        """Queue a robot command on the worker thread."""
        if self.robot is None:
            print("Not connected. Press Connect first.", file=sys.stderr)
            return
        self.pool.start(Task(fn))


def run():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
