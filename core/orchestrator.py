import logging
from typing import Dict

from .event_bus import EventBus
from .module_base import Module

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates all modules via the event bus."""

    def __init__(self, config: dict):
        self.config = config
        self.event_bus = EventBus()
        self._modules: Dict[str, Module] = {}

    def register(self, module: Module, module_config: dict | None = None) -> None:
        cfg = module_config or {}
        module.setup(cfg, self.event_bus)
        self._modules[module.name] = module
        logger.info("Registered module: %s", module.name)

    def start(self) -> None:
        logger.info("Starting orchestrator with %d module(s)", len(self._modules))
        for name, module in self._modules.items():
            if module.enabled:
                logger.info("Starting module: %s", name)
                module.start()

    def stop(self) -> None:
        logger.info("Stopping orchestrator")
        for name, module in reversed(list(self._modules.items())):
            if module.enabled:
                logger.info("Stopping module: %s", name)
                try:
                    module.stop()
                except Exception:
                    logger.exception("Error stopping module: %s", name)
        self.event_bus.emit(EventBus.SHUTDOWN)
