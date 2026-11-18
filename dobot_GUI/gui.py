import os
import sys
import threading
import traceback
from datetime import datetime

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget, QTabWidget, QGridLayout, QComboBox,
    QSizePolicy, QScrollArea
)

from config_manager import CONFIG_PATH, get_float, get_int, get_str, set_value
from dobot_pick import DobotE6
from functions import ( connect_robot, move_to_home, save_current_orientation_as_home, 
    save_work_reference, calibrate_system, run_pick)
from vision import load_roboflow_model


COLOR_WARNING = "#e6a200"
COLOR_ERROR = "#e05555"

STYLE_LOCKED = """
    QLineEdit, QLineEdit:read-only, QComboBox, QComboBox:disabled {
        background-color: #f0f0f0;
        color: #000000;
        border: 2px solid #cccccc;
        padding: 3px;
    }
"""

STYLE_EDIT = """
    QLineEdit, QComboBox {
        background-color: #ffffff;
        color: #000000;
        border: 2px solid #0078d7;
        padding: 3px;
    }
"""

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
        layout.addWidget(self._build_tabs(), stretch=3)
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

        connect_btn = QPushButton("Connect and Enable")
        connect_btn.clicked.connect(self.connect)
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

        save_home_button = QPushButton("Save home")
        save_home_button.clicked.connect(lambda: self.run_robot(lambda: save_current_orientation_as_home(self.robot)))
        #

        save_work_reference_button = QPushButton("Save reference")
        save_work_reference_button.clicked.connect(lambda: self.run_robot(lambda: save_work_reference(self.robot)))
        #
        calibrate_button = QPushButton("Calibrate")
        calibrate_button.clicked.connect(self.calibrate)

        pick_button = QPushButton("Pick one")
        pick_button.clicked.connect(lambda: self.run_robot(lambda: run_pick(self.robot, self.model, self.mm_per_pixel, self.marker_corners)))

        estop_btn = QPushButton("EMERGENCY STOP")
        estop_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; border-radius: 10px; padding:5px")
        estop_btn.clicked.connect(self.emergency_stop)

        for btn in (enable_btn, clear_btn, home_btn, save_home_button, save_work_reference_button, calibrate_button, pick_button):
            row.addWidget(btn)
        row.addStretch()
        row.addWidget(estop_btn)
        return box

    # TESTING // Alexander
    def _build_tabs(self):
        tabs = QTabWidget()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)

        schema = {
            "Camera": [
                {"key": "USE_WEBCAM", "options": ["0", "1"]},
                {"key": "OUTPUT_DIR", "max_len": 100},
                {"key": "PREVIEW_WIDTH", "max_len": 10},
                {"key": "PREVIEW_HEIGHT", "max_len": 10},
            ],
            "Model": [
                {"key": "ROBOFLOW_API_KEY", "max_len": 50},
                {"key": "ROBOFLOW_PROJECT", "max_len": 50},
                {"key": "ROBOFLOW_MODEL_VERSION", "max_len": 10},
                {"key": "CONFIDENCE_THRESHOLD", "max_len": 10},
            ],
            "Calibration": [
                {"key": "MARKER_SIZE_MM", "max_len": 15},
                {"key": "ARUCO_DICT", "options": ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_ARUCO_ORIGINAL"]},
            ],
            "Pick": [
                {"key": "PICK_Z", "max_len": 15},
                {"key": "SUCTION_DELAY", "max_len": 15},
                {"key": "X_OFFSET", "max_len": 15},
                {"key": "Y_OFFSET", "max_len": 15},
                {"key": "Z_OFFSET", "max_len": 15},
            ],
            "Robot Connection": [
                {"key": "ROBOT_SPEED", "max_len": 10},
                {"key": "USER_COORD_INDEX", "max_len": 10},
                {"key": "TOOL_INDEX", "max_len": 10},
            ],
            "Robot Position": [
                {"key": "VERIFY_Z", "max_len": 15},
                {"key": "SAFE_Z", "max_len": 15},
                {"key": "HOME_X", "max_len": 15},
                {"key": "HOME_Y", "max_len": 15},
                {"key": "HOME_Z", "max_len": 15},
                {"key": "HOME_RX", "max_len": 15},
                {"key": "HOME_RY", "max_len": 15},
                {"key": "HOME_RZ", "max_len": 15},
                {"key": "DROP_X", "max_len": 15},
                {"key": "DROP_Y", "max_len": 15},
                {"key": "DROP_Z", "max_len": 15},
            ],
            "Safety Limits": [
                {"key": "WORKSPACE_X_MIN", "max_len": 15},
                {"key": "WORKSPACE_X_MAX", "max_len": 15},
                {"key": "WORKSPACE_Y_MIN", "max_len": 15},
                {"key": "WORKSPACE_Y_MAX", "max_len": 15},
            ],
            "Debug Options": [
                {"key": "SIMULATE_ROBOT", "options": ["0", "1"]},
                {"key": "TEST_IMAGE", "max_len": 100},
                {"key": "SAVE_DEBUG_IMAGES", "options": ["0", "1"]},
            ]
        }

        for section, fields in schema.items():
            self._add_category(layout, section, fields, cols=5)

        layout.addStretch(1)
        scroll.setWidget(content)

        tab2 = QWidget()
        QVBoxLayout(tab2).addWidget(QLabel("Step 1......."))

        tabs.addTab(scroll, "Config panel")
        tabs.addTab(tab2, "Guide")
        return tabs

    def _add_category(self, layout, section, fields, cols=5):
        cat_box = QVBoxLayout()
        cat_box.setSpacing(8)

        header_row = QHBoxLayout()
        header = QLabel(section)
        header.setStyleSheet("font-weight: bold; font-size: 18px; margin-top: 8px;")
        header_row.addWidget(header)
        header_row.addStretch(1)

        btn = QPushButton("Edit")
        btn.setFixedWidth(75)
        header_row.addWidget(btn)
        cat_box.addLayout(header_row)

        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #cccccc;")
        cat_box.addWidget(line)

        section_widgets = []
        for i in range(0, len(fields), cols):
            row_box = QHBoxLayout()
            row_box.setSpacing(10)
            chunk = fields[i:i + cols]

            for f in chunk:
                key = f["key"]
                val = get_str(section, key, fallback="")

                box = QVBoxLayout()
                box.setSpacing(2)
                box.addWidget(QLabel(key + ":"))

                if "options" in f:
                    w = QComboBox()
                    w.addItems(f["options"])
                    w.setCurrentText(val)
                    w.setEnabled(False)
                else:
                    w = QLineEdit()
                    w.setMaxLength(f.get("max_len", 30))
                    w.setText(val)
                    w.setReadOnly(True)

                w.setStyleSheet(STYLE_LOCKED)
                w.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
                box.addWidget(w)

                section_widgets.append((section, key, w))
                row_box.addLayout(box, stretch=1)

            for _ in range(cols - len(chunk)):
                row_box.addStretch(1)

            cat_box.addLayout(row_box)

        btn.clicked.connect(lambda: self.toggle_edit(section_widgets, btn))
        layout.addLayout(cat_box)

    def toggle_edit(self, section_widgets, btn):
        first = section_widgets[0][2]
        locked = first.isReadOnly() if isinstance(first, QLineEdit) else not first.isEnabled()

        for section, key, w in section_widgets:
            if isinstance(w, QLineEdit):
                w.setReadOnly(not locked)
                val = w.text()
            else:
                w.setEnabled(locked)
                val = w.currentText()

            w.setStyleSheet(STYLE_EDIT if locked else STYLE_LOCKED)

            if not locked:
                set_value(section, key, val)

        if locked:
            first.setFocus()
            btn.setText("Save")
        else:
            btn.setText("Edit")


    def connect(self):
        try:
            port = int(self.port_edit.text())
        except ValueError:
            print("ERROR: Port must be a number.", file=sys.stderr)
            return
        ip = self.ip_edit.text().strip() or None
        def task():
            robot = connect_robot(ip, port)
            if robot is not None:
                self.robot = robot
        self.pool.start(Task(task))
        if hasattr(self, "model"):
            print("Model already loaded")
        else:
            self.load_AI_model()



    def disconnect_robot(self):
        if self.robot is not None:
            self.run_robot(self.robot.disconnect)

    def go_home(self):
        self.run_robot(lambda: move_to_home(self.robot))

    def calibrate(self):
        def task():
            mm_per_pixel, marker_corners = calibrate_system(self.robot)
            if mm_per_pixel is not None and marker_corners is not None:
                self.mm_per_pixel = mm_per_pixel
                self.marker_corners = marker_corners
        self.run_robot(task)

    def load_AI_model(self):
        def task():
            model = load_roboflow_model()
            if model is not None:
                self.model = model
                print("Model load successfully!")
        self.run_robot(task)

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
