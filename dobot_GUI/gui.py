import os
import sys
import threading
import traceback
from datetime import datetime
import numpy as np
import cv2

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget, QTabWidget, QGridLayout, QComboBox,
    QSizePolicy, QScrollArea, QProgressDialog
)

from config_manager import CONFIG_PATH, get_float, get_int, get_str, set_value
from dobot_pick import DobotE6
from functions import ( connect_robot, move_to_home, save_current_orientation_as_home, 
    save_reference_c0, save_reference_c1, calibrate_system, run_pick)
from vision import load_roboflow_model, process_predictions, run_inference


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

class TaskSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
class Task(QRunnable):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = TaskSignals()

    def run(self):
        try:
            result = self.fn()
            self.signals.finished.emit(result)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            self.signals.error.emit(str(exc))
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dobot Pick-and-Place")
        self.showFullScreen()

        self.robot = None
        self.model = None
        self.mm_per_pixel = None
        self.marker_corners = None

        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        self.tab_widget = self._build_tabs()

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
        layout.addWidget(self.tab_widget, stretch=4)
        layout.addWidget(self.log, stretch=1)
        self.setCentralWidget(central)

        if self.log_file is not None:
            print(f"Session log: {self.log_file.name}")
        print("Ready. Enter the robot's IP and port, then press Connect.")

        c0_x = get_float("Calibration", "C0_PX_X", 0.0)
        c0_y = get_float("Calibration", "C0_PX_Y", 0.0)
        c1_x = get_float("Calibration", "C1_PX_X", 0.0)
        c1_y = get_float("Calibration", "C1_PX_Y", 0.0)
        if c0_x != 0.0 or c0_y != 0.0 or c1_x != 0.0 or c1_y != 0.0:
            self.marker_corners = np.array([[c0_x, c0_y], [c1_x, c1_y]], dtype=float)
            print("Loaded saved calibration from config.ini.")

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
        connect_btn.clicked.connect(self.on_connect)
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
        save_home_button.clicked.connect(self.save_home)
        #

        save_c0_button = QPushButton("Save C0")
        save_c0_button.clicked.connect(lambda: self.run_robot(lambda: save_reference_c0(self.robot)))

        save_c1_button = QPushButton("Save C1")
        save_c1_button.clicked.connect(lambda: self.run_robot(lambda: save_reference_c1(self.robot)))
        #
        calibrate_button = QPushButton("Calibrate")
        calibrate_button.clicked.connect(self.calibrate)

        pick_button = QPushButton("Pick one")
        pick_button.clicked.connect(
            lambda: self.run_robot(
                lambda: run_pick(
                    self.robot, 
                    self.model, 
                    self.mm_per_pixel, 
                    self.marker_corners, 
                    image_callback=lambda img: self.display_in_pyside(img, None, self.result_image_label)
                )
            )
        )

        load_model_button = QPushButton("Load model")
        load_model_button.clicked.connect(self.load_AI_model)

        segmentation_button = QPushButton("Segmentation")
        segmentation_button.clicked.connect(self.run_analysis)

        estop_btn = QPushButton("EMERGENCY STOP")
        estop_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; border-radius: 10px; padding:5px")
        estop_btn.clicked.connect(self.emergency_stop)

        for btn in (enable_btn, clear_btn, home_btn, save_home_button, save_c0_button, save_c1_button, calibrate_button, pick_button,
                    segmentation_button, load_model_button):
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
                {"key": "X1_REF",       "max_len": 15},
                {"key": "Y1_REF",       "max_len": 15},
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
        layout2 = QVBoxLayout(tab2)
        layout2.setContentsMargins(0, 0, 0, 0)

        self.result_image_label = QLabel("Segmentation results")
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setMinimumSize(1, 1)
        layout2.addWidget(self.result_image_label)

        tab3 = QWidget()
        layout3 = QVBoxLayout(tab3)
        layout3.setContentsMargins(0, 0, 0, 0)

        self.aruco_image_label = QLabel("Aruco marker detection")
        self.aruco_image_label.setAlignment(Qt.AlignCenter)
        self.aruco_image_label.setMinimumSize(1, 1)
        layout3.addWidget(self.aruco_image_label)

        tabs.addTab(scroll, "Config panel")
        tabs.addTab(tab2, "Segmentation")
        tabs.addTab(tab3, "Calibration")
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

    def display_in_pyside(self, image_bgr, result_json=None, label_widget=None):
        if result_json is not None:
            processed_image, centroids = process_predictions(image_bgr, result_json)
        else:
            processed_image = image_bgr

        image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_image)

        # Save to the matching pixmap variable so they never mix up
        if label_widget is self.aruco_image_label:
            self.aruco_pixmap = pixmap
        else:
            self.result_pixmap = pixmap

        self._rescale_image()

    def _rescale_image(self):
        # Rescale segmentation image if it exists
        if getattr(self, "result_pixmap", None) is not None:
            scaled_result = self.result_pixmap.scaled(
                self.result_image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.result_image_label.setPixmap(scaled_result)

        # Rescale ArUco calibration image if it exists
        if getattr(self, "aruco_pixmap", None) is not None:
            scaled_aruco = self.aruco_pixmap.scaled(
                self.aruco_image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.aruco_image_label.setPixmap(scaled_aruco)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale_image()

    def run_analysis(self, IMG_PATH=None):
        img_path = IMG_PATH if isinstance(IMG_PATH, str) else ""

        if getattr(self, "model", None) is None:
            print("ERROR: Model not loaded yet!")
            return

        self.tab_widget.setCurrentIndex(1)

        self.loading_dialog = QProgressDialog("Segmenting image...", "", 0, 0, self)
        self.loading_dialog.setCancelButton(None)
        self.loading_dialog.setWindowModality(Qt.WindowModal)
        self.loading_dialog.show()

        def task():
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            return image_bgr, run_inference(self.model, img_path)

        self.current_task = Task(task)
        self.current_task.signals.finished.connect(lambda res: (
            self.loading_dialog.close(),
            self.tab_widget.setCurrentIndex(1),
            self.display_in_pyside(res[0], res[1], self.result_image_label)
        ))
        self.current_task.signals.error.connect(lambda err: self.loading_dialog.close())
        self.pool.start(self.current_task)

    def run_calibration(self, IMG_PATH=None):
        img_path = IMG_PATH if isinstance(IMG_PATH, str) else ""
        self.tab_widget.setCurrentIndex(2)

        def task():
            image_bgr = cv2.imread(img_path)
            return calibrate_system(None, image_bgr)

        self.current_task = Task(task)
        self.current_task.signals.finished.connect(lambda res: (
            setattr(self, "mm_per_pixel", res[0]),
            setattr(self, "c0_px", res[1]),
            setattr(self, "marker_corners", res[2]),
            self.display_in_pyside(res[3], None, self.aruco_image_label)
        ))
        self.pool.start(self.current_task)



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


    def on_connect(self):
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
        if self.model is not None:
            print("Model already loaded")
        else:
            self.load_AI_model()

    def disconnect_robot(self):
        if self.robot is not None:
            self.run_robot(self.robot.disconnect)

    def go_home(self):
        self.run_robot(lambda: move_to_home(self.robot))

    def save_home(self):
        def task():
            if save_current_orientation_as_home(self.robot):
                self.marker_corners = None
                print("WARNING: Home-position changed. Run Calibrate (and Save C0/C1 if Aruco-marker has moved).")
        self.run_robot(task)

    def calibrate(self):
        def task():
            mm_per_pixel, c0_px, marker_corners, disp = calibrate_system(self.robot)
            if mm_per_pixel is not None and marker_corners is not None:
                self.display_in_pyside(disp, None, self.aruco_image_label)
                self.mm_per_pixel = mm_per_pixel
                self.marker_corners = marker_corners
        self.pool.start(Task(task)) # fixa tillbaka till self.run_robot(task)

    def load_AI_model(self):
        def task():
            model = load_roboflow_model()
            if model is not None:
                self.model = model
                print("Model load successfully!")
        self.pool.start(Task(task))

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
