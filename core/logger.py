import logging
import json
import sys
from datetime import datetime, timezone

try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.theme import Theme
    _RICH = True
except ImportError:
    _RICH = False

_THEME = Theme({
    "logging.level.info": "bold cyan",
    "logging.level.warning": "bold yellow",
    "logging.level.error": "bold red",
    "logging.level.debug": "dim white"
})

def _make_rich_handler() -> logging.Handler:
    console = Console(stderr=False, theme=_THEME)
    return RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        log_time_format=["%H:%M:%S"]
        markup=True
    )

def _make_plain_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    return handler

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        if hasattr(record, "extra"):
            base.update(record.extra)
        return json.dumps(base)


class StructuredLogger(logging.Logger):
    def _log_structured(self, level, event, **kwargs):
        record = self.makeRecord(self.name, level, "", 0, event, (), None)
        record.extra = kwargs
        self.handle(record)

    def info(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        self._log_structured(logging.INFO, msg, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        self._log_structured(logging.WARNING, msg, **kwargs)

    def error(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        self._log_structured(logging.ERROR, msg, **kwargs)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
