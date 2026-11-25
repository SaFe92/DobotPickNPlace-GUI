import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDockWidget, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from camera_feed import CameraFeed
from config_manager import get_int
from gui import MainWindow

class LiveFeedWindow(MainWindow):
    def __init__(self):
        super().__init__()
        self.camera_feed = None
        dock = QDockWidget("Live feed", self)
        dock.setWidget(self._build_feed_widget())
        dock.visibilityChanged.connect(self._on_dock_visibility)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_feed_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        row = QHBoxLayout()
        start_btn = QPushButton("Start")
        start_btn.clicked.connect(self.start_camera_feed)
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self.stop_camera_feed)
        row.addWidget(start_btn)
        row.addWidget(stop_btn)
        row.addStretch()
        layout.addLayout(row)

        self.feed_label = QLabel("Camera is off")
        self.feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed_label.setFixedSize(
            get_int("Camera", "PREVIEW_WIDTH", 640) // 2,
            get_int("Camera", "PREVIEW_HEIGHT", 480) // 2,
        )
        layout.addWidget(self.feed_label)
        layout.addStretch()
        return widget

    def start_camera_feed(self):
        self.stop_camera_feed()
        self.camera_feed = CameraFeed(parent=self)
        self.camera_feed.frame_ready.connect(self.show_frame)
        self.camera_feed.failed.connect(lambda msg: print(f"ERROR: {msg}", file=sys.stderr))
        self.camera_feed.start()

    def stop_camera_feed(self):
        if self.camera_feed is not None:
            self.camera_feed.stop()
            self.camera_feed.wait(1000)
            self.camera_feed = None
            self.feed_label.setText("Camera is off")

    def show_frame(self, image):
        pixmap = QPixmap.fromImage(image).scaled(
            self.feed_label.size(), Qt.AspectRatioMode.KeepAspectRatio
        )
        self.feed_label.setPixmap(pixmap)

    def _on_dock_visibility(self, visible):
        if not visible:
            self.stop_camera_feed()

    def closeEvent(self, event):
        self.stop_camera_feed()
        super().closeEvent(event)

    def calibrate(self):
        self.stop_camera_feed()
        super().calibrate()

    def run_robot(self, fn):
        self.stop_camera_feed()
        super().run_robot(fn)
