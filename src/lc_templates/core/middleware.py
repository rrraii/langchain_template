from __future__ import annotations

import re
from collections.abc import Iterable

from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    wrap_model_call,
)
from langchain_core.messages import HumanMessage

from lc_templates.core.config import get_settings
from lc_templates.core.hooks import emit_event
from lc_templates.core.logging import get_logger
from lc_templates.core.models import build_chat_model, build_reasoning_model
from lc_templates.core.schemas import ExecutionMetadata

logger = get_logger(__name__)

_PROFILE_FACT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bmy name is\b",
        r"\bi am called\b",
        r"\bi'm\b",
        r"我叫",
        r"我的名字",
        r"请叫我",
        r"记住我",
        r"我的偏好",
        r"我喜欢",
        r"我在学",
    ]
]


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


def _message_text(message: object) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _tokenize_for_similarity(text: str) -> set[str]:
    lowered = text.lower()
    english_tokens = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    chinese_pairs = {
        lowered[idx : idx + 2]
        for idx in range(len(lowered) - 1)
        if re.match(r"[\u4e00-\u9fff]{2}", lowered[idx : idx + 2])
    }
    return english_tokens | chinese_pairs


def _text_similarity(left: str, right: str) -> float:
    left_tokens = _tokenize_for_similarity(left)
    right_tokens = _tokenize_for_similarity(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


def _looks_like_topic_shift(
    current: str,
    previous_messages: Iterable[object],
    threshold: float,
) -> bool:
    current = current.strip()
    if not current:
        return False
    previous_human_texts = [
        _message_text(message).strip()
        for message in previous_messages
        if isinstance(message, HumanMessage) and _message_text(message).strip()
    ]
    if not previous_human_texts:
        return False
    best_similarity = max(_text_similarity(current, previous) for previous in previous_human_texts)
    return best_similarity < threshold


def _is_profile_fact_message(message: object) -> bool:
    if not isinstance(message, HumanMessage):
        return False
    text = _message_text(message)
    return any(pattern.search(text) for pattern in _PROFILE_FACT_PATTERNS)


def _select_messages_for_current_turn(messages: list[object], recent_messages: int) -> list[object]:
    if recent_messages <= 0 or len(messages) <= recent_messages:
        return list(messages)
    return list(messages[-recent_messages:])


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


@wrap_model_call(name="CurrentTurnPriorityMiddleware")
def _current_turn_priority(request, handler):
    settings = get_settings()
    context_guard = settings.runtime.middleware.context_guard
    if not context_guard.enabled or len(request.messages) <= context_guard.recent_messages + 1:
        return handler(request)

    human_messages = [message for message in request.messages if isinstance(message, HumanMessage)]
    if len(human_messages) < 2:
        return handler(request)

    current_message = human_messages[-1]
    previous_messages = request.messages[:-1]
    current_text = _message_text(current_message)
    if not _looks_like_topic_shift(
        current_text,
        previous_messages,
        context_guard.similarity_threshold,
    ):
        return handler(request)

    edited_messages = _select_messages_for_current_turn(
        request.messages,
        context_guard.recent_messages,
    )
    if context_guard.preserve_profile_facts:
        profile_messages = [
            message
            for message in request.messages[:-context_guard.recent_messages]
            if _is_profile_fact_message(message)
        ]
        deduped_profile_messages: list[object] = []
        seen_texts: set[str] = set()
        for message in profile_messages:
            text = _message_text(message).strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                deduped_profile_messages.append(message)
        edited_messages = deduped_profile_messages + edited_messages

    logger.info(
        (
            "Current-turn priority middleware trimmed memory context. "
            "original=%s retained=%s threshold=%s"
        ),
        len(request.messages),
        len(edited_messages),
        context_guard.similarity_threshold,
    )
    emit_event(
        "middleware.current_turn_priority",
        message="Context guard reduced stale memory influence after a topic shift.",
        meta=ExecutionMetadata(
            provider_name=_active_provider_name(settings),
            operation="agent_middleware",
        ),
        payload={
            "original_message_count": len(request.messages),
            "retained_message_count": len(edited_messages),
            "similarity_threshold": context_guard.similarity_threshold,
            "recent_messages": context_guard.recent_messages,
            "preserve_profile_facts": context_guard.preserve_profile_facts,
        },
    )
    return handler(request.override(messages=edited_messages))


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

    if has_memory and config.context_guard.enabled:
        middleware.append(_current_turn_priority)

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
