from __future__ import annotations

import json
import re
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from lc_templates.core.config import get_settings
from lc_templates.core.schemas import (
    AgentExecutionResult,
    ClassificationResult,
    ExtractionResult,
    GroundedAnswer,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)
DetailLevel = Literal["concise", "verbose"]


def extract_text_content(value: Any) -> str:
    if isinstance(value, str):
        return value

    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)

    text_attr = getattr(value, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    return str(value)


def normalize_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def coerce_response_text(value: Any) -> str:
    return normalize_text(extract_text_content(value))


def get_final_message_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    for message in reversed(messages):
        text = coerce_response_text(message)
        if text:
            return text
    return ""


def collect_tool_names(result: dict[str, Any]) -> list[str]:
    executed_tools: list[str] = []
    proposed_tools: list[str] = []
    for message in result.get("messages", []):
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                name = tool_call.get("name")
                if name:
                    proposed_tools.append(name)
        name = getattr(message, "name", "")
        if name:
            executed_tools.append(name)
    return executed_tools or proposed_tools


def build_agent_execution_result(result: dict[str, Any]) -> AgentExecutionResult:
    tool_names = collect_tool_names(result)
    return AgentExecutionResult(
        final_text=get_final_message_text(result),
        used_tools=tool_names,
        tool_call_count=len(tool_names),
        raw=result,
    )


def get_response_format_instructions() -> str:
    settings = get_settings().runtime
    style_map = {
        "concise": "Keep the answer short and focused.",
        "balanced": "Balance brevity with enough explanation to be reliable.",
        "detailed": "Be thorough, but avoid unnecessary repetition.",
    }
    format_map = {
        "markdown": (
            "Respond in clean Markdown. "
            "Use short sections or bullets only when they improve readability."
        ),
        "text": "Respond in plain text without Markdown syntax.",
    }
    return " ".join(
        [
            f"Respond in {settings.response_language}.",
            format_map[settings.response_format],
            style_map[settings.answer_style],
        ]
    )


def render_classification_result(
    result: ClassificationResult, detail_level: DetailLevel = "concise"
) -> str:
    if detail_level == "verbose":
        return "\n".join(
            [
                f"Label: {result.label}",
                f"Confidence: {result.confidence:.2f}",
                f"Reason: {result.reason}",
            ]
        )
    return f"{result.label} ({result.confidence:.2f})"


def render_agent_execution_result(
    result: AgentExecutionResult, detail_level: DetailLevel = "concise"
) -> str:
    final_text = normalize_text(result.final_text)
    if detail_level == "verbose":
        lines = [f"Answer:\n{final_text or '(empty)'}"]
        if result.used_tools:
            lines.append(f"Tools: {', '.join(result.used_tools)}")
        lines.append(f"Tool call count: {result.tool_call_count}")
        return "\n\n".join(lines)

    if result.used_tools:
        return f"{final_text}\n\nTools: {', '.join(result.used_tools)}"
    return final_text


def render_extraction_result(
    result: ExtractionResult, detail_level: DetailLevel = "concise"
) -> str:
    entities = [item for item in result.entities if item]
    if detail_level == "verbose":
        entity_lines = "\n".join(f"- {item}" for item in entities) if entities else "- (none)"
        lines = [
            f"Status: {result.status}",
            f"Fallback used: {result.fallback_used}",
            f"Summary: {result.summary or '(empty)'}",
            "Entities:",
            entity_lines,
        ]
        if result.error_reason:
            lines.append(f"Error reason: {result.error_reason}")
        return "\n".join(lines)

    if not result.summary and not entities:
        return "No entities extracted."

    if entities:
        if result.summary:
            return f"{result.summary}\nEntities: {', '.join(entities)}"
        return f"Entities: {', '.join(entities)}"
    return result.summary


def render_route_result(route: str, detail_level: DetailLevel = "concise") -> str:
    normalized = normalize_text(route)
    if detail_level == "verbose":
        return normalized
    if normalized.startswith("route:"):
        return normalized.split(":", 1)[1]
    return normalized


def render_task_result(result: dict[str, Any], detail_level: DetailLevel = "concise") -> str:
    if detail_level == "verbose":
        return json.dumps(result, ensure_ascii=False, indent=2)

    summary = normalize_text(str(result.get("summary", "")))
    classification = result.get("classification") or {}
    extraction = result.get("extraction") or {}
    label = str(classification.get("label", "")).strip()
    extraction_summary = normalize_text(str(extraction.get("summary", "")))

    route_hint = f"Route hint: {label}" if label else ""
    parts = [part for part in [summary, route_hint, extraction_summary] if part]
    return "\n".join(parts)


def render_citation_answer(answer: str, citations: list[str]) -> str:
    settings = get_settings().runtime
    answer = normalize_text(answer)
    citations = [normalize_text(item) for item in citations if normalize_text(item)]
    citations = citations[: settings.max_citations]

    if not citations:
        return answer

    if settings.response_format == "markdown":
        citation_lines = "\n".join(f"- {item}" for item in citations)
        return f"{answer}\n\nReferences\n{citation_lines}"

    citation_lines = "\n".join(f"* {item}" for item in citations)
    return f"{answer}\n\nReferences:\n{citation_lines}"


def render_grounded_answer(result: GroundedAnswer) -> str:
    settings = get_settings().runtime
    if not result.grounded:
        return settings.rag_no_answer_message
    return render_citation_answer(result.answer, result.citations)


def strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_json_model(text: str, schema: type[SchemaT]) -> SchemaT:
    stripped = strip_json_fences(text)
    try:
        return schema.model_validate_json(stripped)
    except Exception:
        return schema.model_validate(json.loads(stripped))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    return value


def to_pretty_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, indent=2)
