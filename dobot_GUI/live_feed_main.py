import sys

from PySide6.QtWidgets import QApplication

from live_feed_window import LiveFeedWindow

# This file does what gui.run() does, but starts LiveFeedWindow instead of
# MainWindow. It exists because main.py can't be changed: keeping the feed in its
# own entry point leaves the team's startup path untouched.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveFeedWindow()
    window.show()
    sys.exit(app.exec())
