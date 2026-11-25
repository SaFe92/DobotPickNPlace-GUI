import time
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

TIMEOUT_MS = 200
TIMELIMIT_WITHOUT_FRAMES = 5.0

class CameraFeed(QThread):
    frame_ready = Signal(QImage)
    failed = Signal(str)

    def __init__(self, max_fps=15, parent=None):
        super().__init__(parent)
        self._frame_interval = 1.0 / max_fps
        self._running = True  

    def stop(self):
        self._running = False

    def run(self):
        try:
            from pypylon import pylon
        except ImportError:
            self.failed.emit("pypylon is not installed")
            return

        camera = None
        try:
            factory = pylon.TlFactory.GetInstance()
            devices = factory.EnumerateDevices()
            if not devices:
                raise RuntimeError("No camera found")
            camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
            camera.Open()

            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            latest_frame_time = time.monotonic() # we run this monotonic to count - since a counter doesn't account for FPS variations and a clock could be changed mid-program. Was suggested in documentation. 
            next_frame_time = 0.0
            while self._running:
                res = camera.RetrieveResult(TIMEOUT_MS, pylon.TimeoutHandling_Return)
                try:
                    if res.GrabSucceeded():
                        now = time.monotonic()
                        latest_frame_time = now
                        if now >= next_frame_time:
                            next_frame_time = now + self._frame_interval
                            bgr = converter.Convert(res).GetArray()
                            rgb = np.ascontiguousarray(bgr[:, :, ::-1]) # convert from baslers raw format to bgr - then channel swap by reading the array backwards (theoretically this should work, but idk), because QImage uses RGB.
                            height = rgb.shape[0]
                            width = rgb.shape[1]
                            data = rgb.data
                            row_bytes = rgb.strides[0]
                            qformat = QImage.Format.Format_RGB888

                            image = QImage(data, width, height, row_bytes, qformat)
                            image = image.copy()
                            self.frame_ready.emit(image)
                    elif time.monotonic() - latest_frame_time > TIMELIMIT_WITHOUT_FRAMES:
                        raise RuntimeError(f"Camera timed out")
                finally:
                    res.Release()   # buffer needs to be released 
        except Exception as exc:
            self.failed.emit(f"Camera error: {exc}")
        finally:
            if camera is not None:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
                if camera.IsOpen():
                    camera.Close()


