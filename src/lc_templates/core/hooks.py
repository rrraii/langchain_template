from __future__ import annotations

from collections.abc import Callable
from threading import RLock

from lc_templates.core.schemas import ExecutionMetadata, HookEvent

EventHook = Callable[[HookEvent], None]

_HOOKS: list[EventHook] = []
_LOCK = RLock()


def register_event_hook(callback: EventHook) -> EventHook:
    with _LOCK:
        _HOOKS.append(callback)
    return callback


def unregister_event_hook(callback: EventHook) -> None:
    with _LOCK:
        _HOOKS[:] = [hook for hook in _HOOKS if hook is not callback]


def clear_event_hooks() -> None:
    with _LOCK:
        _HOOKS.clear()


def list_event_hooks() -> list[EventHook]:
    with _LOCK:
        return list(_HOOKS)


def emit_event(
    name: str,
    *,
    level: str = "INFO",
    message: str = "",
    trace_id: str = "",
    meta: ExecutionMetadata | None = None,
    payload: dict | None = None,
) -> HookEvent:
    event = HookEvent(
        name=name,
        level=level,
        message=message,
        trace_id=trace_id,
        meta=meta or ExecutionMetadata(),
        payload=dict(payload or {}),
    )
    for hook in list_event_hooks():
        hook(event)
    return event
