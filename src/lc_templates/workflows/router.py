from lc_templates.chains.structured_output import classify_text
from lc_templates.core.config import get_settings
from lc_templates.core.schemas import ExecutionMetadata, RouteDecision

ROUTE_LABELS = ["rag", "extract", "summarize", "chat"]


def route_decision(text: str) -> RouteDecision:
    result = classify_text(text, ROUTE_LABELS)
    threshold = get_settings().runtime.routing_confidence_threshold
    meta = ExecutionMetadata(
        provider_name=result.meta.provider_name,
        model_name=result.meta.model_name,
        operation="route",
    )
    if result.confidence < threshold:
        return RouteDecision(
            route="route:chat",
            name="chat",
            confidence=result.confidence,
            reason=result.reason or "Confidence below routing threshold.",
            status="partial",
            fallback_used=True,
            error_reason="confidence_below_threshold",
            meta=meta,
        )

    label = result.label if result.label in ROUTE_LABELS else "chat"
    return RouteDecision(
        route=f"route:{label}",
        name=label,
        confidence=result.confidence,
        reason=result.reason,
        meta=meta,
    )


def route_task(text: str) -> str:
    return route_decision(text).route
