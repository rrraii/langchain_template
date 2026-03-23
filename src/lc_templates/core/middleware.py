from __future__ import annotations

from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    wrap_model_call,
)

from lc_templates.core.config import get_settings
from lc_templates.core.hooks import emit_event
from lc_templates.core.logging import get_logger
from lc_templates.core.models import build_chat_model, build_reasoning_model
from lc_templates.core.schemas import ExecutionMetadata

logger = get_logger(__name__)


def _active_provider_name(settings) -> str:
    getter = getattr(settings, "get_active_provider_name", None)
    if callable(getter):
        return str(getter())
    return ""


def _message_text_length(messages: list[object]) -> int:
    total = 0
    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    total += len(str(item.get("text", "")))
                elif isinstance(item, str):
                    total += len(item)
    return total


@wrap_model_call(name="DynamicModelSelectionMiddleware")
def _dynamic_model_selection(request, handler):
    settings = get_settings()
    config = settings.runtime.middleware
    provider = settings.get_active_provider()
    fallback_model_name = provider.reasoning_model or provider.chat_model
    if (
        config.dynamic_model_selection_enabled
        and fallback_model_name
        and fallback_model_name != provider.chat_model
        and _message_text_length(request.messages)
        >= config.dynamic_model_selection_message_threshold
    ):
        logger.info(
            "Dynamic model selection switched request to reasoning model. threshold=%s",
            config.dynamic_model_selection_message_threshold,
        )
        emit_event(
            "middleware.dynamic_model_selection",
            message="Dynamic model selection switched to the reasoning model.",
            meta=ExecutionMetadata(
                provider_name=_active_provider_name(settings),
                model_name=fallback_model_name,
                operation="agent_middleware",
            ),
            payload={
                "threshold": config.dynamic_model_selection_message_threshold,
                "message_length": _message_text_length(request.messages),
            },
        )
        request = request.override(model=build_reasoning_model())
    return handler(request)


def build_agent_middleware(*, has_memory: bool = False) -> list[object]:
    settings = get_settings()
    config = settings.runtime.middleware
    if not config.enabled:
        logger.info("Agent middleware disabled by runtime.middleware.enabled=false")
        return []

    middleware: list[object] = []

    if config.dynamic_model_selection_enabled:
        middleware.append(_dynamic_model_selection)

    if config.tool_call_limit_enabled:
        middleware.append(ToolCallLimitMiddleware(run_limit=config.tool_call_limit))

    pii = config.pii
    if pii.enabled:
        for pii_type in pii.pii_types:
            middleware.append(
                PIIMiddleware(
                    pii_type,
                    strategy=pii.strategy,
                    apply_to_input=pii.apply_to_input,
                    apply_to_output=pii.apply_to_output,
                    apply_to_tool_results=pii.apply_to_tool_results,
                )
            )

    if has_memory and config.summarization.enabled:
        middleware.append(
            SummarizationMiddleware(
                model=build_chat_model(),
                trigger=("messages", config.summarization.trigger_messages),
                keep=("messages", config.summarization.keep_messages),
            )
        )

    if config.model_fallback_enabled:
        provider = settings.get_active_provider()
        fallback_model_name = provider.reasoning_model or provider.chat_model
        if fallback_model_name and fallback_model_name != provider.chat_model:
            middleware.append(
                ModelFallbackMiddleware(
                    build_chat_model(),
                    build_reasoning_model(),
                )
            )

    logger.info(
        "Built agent middleware. profile=%s has_memory=%s middleware=%s",
        getattr(config, "profile", "custom"),
        has_memory,
        [item.__class__.__name__ for item in middleware],
    )
    emit_event(
        "middleware.built",
        message="Agent middleware assembled.",
        meta=ExecutionMetadata(
            provider_name=_active_provider_name(settings),
            operation="agent_middleware",
        ),
        payload={
            "profile": getattr(config, "profile", "custom"),
            "has_memory": has_memory,
            "middleware": [item.__class__.__name__ for item in middleware],
        },
    )
    return middleware
