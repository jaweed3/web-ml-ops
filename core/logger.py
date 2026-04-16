import json
import logging
import os
import sys
from datetime import datetime, timezone

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme

    _RICH = True
except ImportError:
    _RICH = False

_THEME = Theme(
    {
        "logging.level.info": "bold cyan",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.debug": "dim white",
    }
)


def _make_rich_handler() -> logging.Handler:
    console = Console(stderr=False, theme=_THEME)
    return RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        log_time_format="[%H:%M:%S]",
        markup=True,
    )


def _make_plain_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
    )
    return handler


class StructuredLogger(logging.Logger):
    def _log_structured(self, level, event, **kwargs):
        extra = "  ".join(f"[dim]{k}[/dim]=[cyan]{v}[/cyan]" for k, v in kwargs.items())
        msg = f"{event}" + (f"{extra}" if extra else "")
        record = self.makeRecord(self.name, level, "", 0, msg, (), None)
        self.handle(record)

    def info(self, msg, *args, **kwargs):
        extra = kwargs.pop("extra", {})
        if args:
            msg = msg % args
        self._log_structured(logging.INFO, msg, **extra)

    def warning(self, msg, *args, **kwargs):
        extra = kwargs.pop("extra", {})
        if args:
            msg = msg % args
        self._log_structured(logging.WARNING, msg, **extra)

    def error(self, msg, *args, **kwargs):
        extra = kwargs.pop("extra", {})
        if args:
            msg = msg % args
        self._log_structured(logging.ERROR, msg, **extra)


_USE_JSON = os.getenv("LOG_FORMAT", "").lower == "json"


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamps": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        if hasattr(record, "extra"):
            base.update(record.extra)
        return json.dumps(base)


def get_logger(name: str) -> StructuredLogger:
    logging.setLoggerClass(StructuredLogger)
    logger = logging.getLogger(name)
    logger.propagate = False

    if not logger.handlers:
        if _USE_JSON:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(JSONFormatter())
        elif _RICH:
            h = _make_rich_handler()
        else:
            h = _make_plain_handler()
        logger.addHandler(h)
        logger.setLevel(logging.INFO)

    return logger
