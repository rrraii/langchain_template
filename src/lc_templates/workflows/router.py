from lc_templates.chains.structured_output import classify_text
from lc_templates.core.config import get_settings

ROUTE_LABELS = ["rag", "extract", "summarize", "chat"]


def route_task(text: str) -> str:
    result = classify_text(text, ROUTE_LABELS)
    threshold = get_settings().runtime.routing_confidence_threshold
    if result.confidence < threshold:
        return "route:chat"
    label = result.label

    if label == "rag":
        return "route:rag"
    if label == "extract":
        return "route:extract"
    if label == "summarize":
        return "route:summarize"
    return "route:chat"
