import threading
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class EventBus:
    """Simple thread-safe publish/subscribe event bus for inter-module communication."""

    # Core events emitted by built-in modules
    SPEECH_DETECTED = "speech.detected"       # data: str (transcribed text)
    RESPONSE_READY = "ai.response_ready"      # data: str (full response text)
    RESPONSE_TOKEN = "ai.response_token"      # data: str (streaming token)
    RESPONSE_START = "ai.response_start"      # data: None
    RESPONSE_END = "ai.response_end"          # data: str (full text)
    SPEAKING_START = "audio.speaking_start"   # data: None
    SPEAKING_END = "audio.speaking_end"       # data: None
    CAMERA_FRAME = "vision.frame"             # data: np.ndarray
    FACE_DETECTED = "vision.face_detected"    # data: dict (future)
    DRIVE_COMMAND = "drive.command"           # data: dict (future)
    SHUTDOWN = "system.shutdown"              # data: None

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, handler: Callable) -> None:
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    h for h in self._subscribers[event_type] if h != handler
                ]

    def emit(self, event_type: str, data: Any = None, source: str = None) -> None:
        logger.debug("Event '%s' from '%s'", event_type, source or "unknown")
        with self._lock:
            handlers = self._subscribers.get(event_type, [])[:]
        for handler in handlers:
            try:
                handler(data)
            except Exception:
                logger.exception("Handler error for event '%s'", event_type)
