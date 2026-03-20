from __future__ import annotations

import logging
from pathlib import Path

from lc_templates.core.config import resolve_runtime_path

_CONFIGURED = False


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def setup_logging(
    *,
    level: str | int = logging.INFO,
    log_file: str | None = None,
    force: bool = False,
) -> None:
    global _CONFIGURED

    if _CONFIGURED and not force:
        return

    resolved_level = _coerce_level(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    root_logger = logging.getLogger()
    if force:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    root_logger.setLevel(resolved_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        target_path = resolve_runtime_path(log_file) or Path(log_file)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target_path, encoding="utf-8")
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _CONFIGURED = True


def configure_logging_from_runtime(runtime) -> None:
    setup_logging(level=runtime.log_level, log_file=runtime.log_file, force=True)
    third_party_level = _coerce_level(runtime.third_party_log_level)
    for logger_name in [
        "openai",
        "openai._base_client",
        "httpx",
        "httpcore",
        "chromadb",
    ]:
        logging.getLogger(logger_name).setLevel(third_party_level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
