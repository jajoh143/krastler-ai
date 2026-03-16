import logging
import sys


def setup(level: str = "INFO") -> None:
    """Configure root logger with a clean, readable format."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s"
    date_fmt = "%H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    root = logging.getLogger()
    root.setLevel(numeric)
    root.handlers = [handler]

    # Quieten noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
