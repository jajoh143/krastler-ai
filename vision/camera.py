"""
Vision / camera module.

Current state: scaffold only — captures frames and emits them on the event bus.
Future integrations (wired to EventBus events):
  - vision.face_detected  → facial recognition (e.g. face_recognition library)
  - vision.object_detected → object detection (e.g. YOLOv8-nano or TFLite)

To enable: set vision.enabled: true in config.yaml and install opencv-python.
"""

import logging
import threading
import time
from typing import Optional

from core.event_bus import EventBus
from core.module_base import Module

logger = logging.getLogger(__name__)


class CameraModule(Module):
    """
    Captures frames from a camera and emits them on the event bus.
    Other modules can subscribe to EventBus.CAMERA_FRAME to consume frames.
    """

    def __init__(self):
        super().__init__("vision")
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._camera_index: int = 0
        self._fps: float = 15.0

    def _setup(self) -> None:
        self._camera_index = self._config.get("camera_index", 0)
        self._fps = self._config.get("fps", 15.0)

    def start(self) -> None:
        try:
            import cv2
        except ImportError:
            logger.error(
                "opencv-python not installed. Install it with: "
                "pip install opencv-python-headless"
            )
            self.enabled = False
            return

        import cv2

        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            logger.error("Could not open camera index %d", self._camera_index)
            self.enabled = False
            return

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="vision-camera")
        self._thread.start()
        logger.info("Camera started (index=%d, fps=%.1f)", self._camera_index, self._fps)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        logger.info("Camera stopped")

    # ------------------------------------------------------------------
    # Capture loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        import cv2

        interval = 1.0 / self._fps
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                self.emit(EventBus.CAMERA_FRAME, frame)
            time.sleep(interval)

    # ------------------------------------------------------------------
    # Extension points (subclass or monkey-patch for facial recognition etc.)
    # ------------------------------------------------------------------

    def on_frame(self, frame) -> None:
        """
        Called for every captured frame. Override or extend to add vision
        processing (facial recognition, object detection, etc.).

        Example future usage:
            faces = face_recognizer.find(frame)
            if faces:
                self.emit(EventBus.FACE_DETECTED, faces)
        """
        pass
