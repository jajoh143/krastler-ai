from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .event_bus import EventBus


class Module(ABC):
    """Base class for all pluggable modules (audio, vision, driving, etc.)."""

    def __init__(self, name: str):
        self.name = name
        self.enabled = False
        self._event_bus: "EventBus | None" = None
        self._config: dict = {}

    def setup(self, config: dict, event_bus: "EventBus") -> None:
        self._config = config
        self._event_bus = event_bus
        self.enabled = True
        self._setup()

    def _setup(self) -> None:
        """Override for module-specific initialization logic."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start the module (called after all modules are set up)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the module and release resources."""

    def emit(self, event_type: str, data=None) -> None:
        if self._event_bus:
            self._event_bus.emit(event_type, data, source=self.name)

    def subscribe(self, event_type: str, handler) -> None:
        if self._event_bus:
            self._event_bus.subscribe(event_type, handler)
