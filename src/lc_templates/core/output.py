from __future__ import annotations

import json
import re
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from lc_templates.core.config import get_settings
from lc_templates.core.schemas import (
    AgentExecutionResult,
    ClassificationResult,
    ExecutionMetadata,
    ExtractionResult,
    GroundedAnswer,
    HealthReport,
    KnowledgeBaseBuildResult,
    RouteDecision,
    TaskBundleResult,
    ToolCallRecord,
    WorkflowExecutionResult,
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


def collect_tool_records(result: dict[str, Any]) -> list[ToolCallRecord]:
    executed_tool_records: list[ToolCallRecord] = []
    proposed_tool_records: list[ToolCallRecord] = []
    for message in result.get("messages", []):
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                name = tool_call.get("name")
                if name:
                    proposed_tool_records.append(
                        ToolCallRecord(
                            name=name,
                            call_id=str(tool_call.get("id", "")),
                            arguments=dict(tool_call.get("args", {}) or {}),
                        )
                    )
        name = getattr(message, "name", "")
        if name:
            executed_tool_records.append(ToolCallRecord(name=name))
    return executed_tool_records or proposed_tool_records


def extract_execution_metadata(result: dict[str, Any], operation: str = "") -> ExecutionMetadata:
    settings = get_settings()
    provider_name = settings.get_active_provider_name()
    model_name = ""
    for message in reversed(result.get("messages", [])):
        response_metadata = getattr(message, "response_metadata", {}) or {}
        model_name = str(
            response_metadata.get("model_name") or response_metadata.get("model") or ""
        ).strip()
        if model_name:
            break
    if not model_name:
        model_name = settings.get_active_provider().chat_model
    return ExecutionMetadata(
        provider_name=provider_name,
        model_name=model_name,
        operation=operation,
    )


def build_agent_execution_result(result: dict[str, Any]) -> AgentExecutionResult:
    tool_records = collect_tool_records(result)
    tool_names = [record.name for record in tool_records if record.name]
    return AgentExecutionResult(
        final_text=get_final_message_text(result),
        used_tools=tool_names,
        tool_calls=tool_records,
        tool_call_count=len(tool_names),
        raw=result,
        meta=extract_execution_metadata(result, operation="agent"),
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
        if result.trace_id:
            lines.append(f"Trace id: {result.trace_id}")
        lines.append(f"Latency (ms): {result.latency_ms:.3f}")
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
        if result.trace_id:
            lines.append(f"Trace id: {result.trace_id}")
        lines.append(f"Latency (ms): {result.latency_ms:.3f}")
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


def render_task_result(
    result: TaskBundleResult | dict[str, Any], detail_level: DetailLevel = "concise"
) -> str:
    if isinstance(result, BaseModel):
        payload = result.model_dump()
    else:
        payload = result

    if detail_level == "verbose":
        return json.dumps(payload, ensure_ascii=False, indent=2)

    summary = normalize_text(str(payload.get("summary", "")))
    route = payload.get("route") or {}
    classification = payload.get("classification") or {}
    extraction = payload.get("extraction") or {}
    label = str(route.get("name", "") or classification.get("label", "")).strip()
    extraction_summary = normalize_text(str(extraction.get("summary", "")))

    route_hint = f"Route hint: {label}" if label else ""
    parts = [part for part in [summary, route_hint, extraction_summary] if part]
    return "\n".join(parts)


def render_workflow_execution_result(
    result: WorkflowExecutionResult, detail_level: DetailLevel = "concise"
) -> str:
    if detail_level == "verbose":
        lines = [
            f"Route: {result.route.route}",
            f"Status: {result.status}",
            f"Fallback used: {result.fallback_used}",
            f"Result:\n{normalize_text(result.text) or '(empty)'}",
        ]
        if result.trace_id:
            lines.append(f"Trace id: {result.trace_id}")
        lines.append(f"Latency (ms): {result.latency_ms:.3f}")
        if result.error_reason:
            lines.append(f"Error reason: {result.error_reason}")
        return "\n\n".join(lines)
    return normalize_text(result.text)


def render_route_decision(
    result: RouteDecision, detail_level: DetailLevel = "concise"
) -> str:
    if detail_level == "verbose":
        lines = [
            f"Route: {result.route}",
            f"Name: {result.name}",
            f"Confidence: {result.confidence:.2f}",
            f"Status: {result.status}",
            f"Fallback used: {result.fallback_used}",
        ]
        if result.trace_id:
            lines.append(f"Trace id: {result.trace_id}")
        lines.append(f"Latency (ms): {result.latency_ms:.3f}")
        if result.reason:
            lines.append(f"Reason: {result.reason}")
        if result.error_reason:
            lines.append(f"Error reason: {result.error_reason}")
        return "\n".join(lines)
    return result.name


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


def render_health_report(result: HealthReport, detail_level: DetailLevel = "concise") -> str:
    if detail_level == "verbose":
        provider_lines = [
            f"- {item.name}: ready={item.ready}, model={item.chat_model or '(unset)'}"
            for item in result.providers
        ]
        warning_lines = [f"- {item.code}: {item.message}" for item in result.warnings]
        recommendation_lines = [f"- {item}" for item in result.recommendations]
        sections = [
            f"Ready: {result.ready}",
            f"Active provider: {result.active_provider}",
            f"Config path: {result.config_path}",
            f"Recommended middleware profile: {result.recommended_middleware_profile}",
            "Providers:",
            "\n".join(provider_lines) if provider_lines else "- (none)",
            "Warnings:",
            "\n".join(warning_lines) if warning_lines else "- (none)",
            "Recommendations:",
            "\n".join(recommendation_lines) if recommendation_lines else "- (none)",
        ]
        if result.recommended_middleware_reason:
            sections.extend(
                [
                    "Recommended profile reason:",
                    result.recommended_middleware_reason,
                ]
            )
        return "\n".join(sections)

    if result.ready:
        provider_ready_text = (
            f"{result.summary.ready_provider_count}/{result.summary.provider_count}"
        )
        return (
            f"Ready. Active provider: {result.active_provider}. "
            f"Providers ready: {provider_ready_text}."
        )
    return (
        f"Not ready. Active provider: {result.active_provider}. "
        f"Warnings: {result.summary.warning_count}."
    )


def render_knowledge_base_result(
    result: KnowledgeBaseBuildResult, detail_level: DetailLevel = "concise"
) -> str:
    if detail_level == "verbose":
        return "\n".join(
            [
                f"Source: {result.source_path}",
                f"Persist directory: {result.persist_directory}",
                f"Collection: {result.collection_name}",
                f"Provider: {result.provider_name}",
                f"Embedding model: {result.embedding_model}",
                f"Embedding dimensions: {result.embedding_dimensions}",
                f"Documents: {result.document_count}",
                f"Chunks: {result.chunk_count}",
                f"Trace id: {result.trace_id}",
                f"Latency (ms): {result.latency_ms:.3f}",
            ]
        )
    return (
        f"Indexed {result.document_count} document(s) into {result.collection_name} "
        f"with {result.chunk_count} chunk(s)."
    )


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
