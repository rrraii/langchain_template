from lc_templates.chains.prompt_chain import summarize_text
from lc_templates.chains.structured_output import classify_text, extract_entities
from lc_templates.core.hooks import emit_event
from lc_templates.core.logging import get_logger
from lc_templates.core.schemas import ExecutionMetadata, TaskBundleResult
from lc_templates.workflows.router import route_decision

TASK_LABELS = ["medical", "legal", "customer_service", "general"]
logger = get_logger(__name__)


def _task_meta(classification, route) -> ExecutionMetadata:
    return ExecutionMetadata(
        provider_name=classification.meta.provider_name or route.meta.provider_name,
        model_name=classification.meta.model_name or route.meta.model_name,
        operation="tasks",
    )


def run_text_tasks(text: str) -> TaskBundleResult:
    logger.info("Running text task bundle.")
    emit_event(
        "tasks.started",
        message="Text task bundle started.",
        meta=ExecutionMetadata(operation="tasks"),
        payload={"input_length": len(text)},
    )
    summary = summarize_text(text)
    classification = classify_text(text, TASK_LABELS)
    extraction = extract_entities(text)
    route = route_decision(text)
    result = TaskBundleResult(
        summary=summary,
        classification=classification,
        extraction=extraction,
        route=route,
        fallback_used=route.fallback_used or extraction.fallback_used,
        status="partial" if route.fallback_used or extraction.fallback_used else "ok",
        meta=_task_meta(classification, route),
    )
    logger.info(
        "Text task bundle completed. classification=%s route=%s extraction_entities=%s",
        result.classification.label,
        result.route.name,
        len(result.extraction.entities),
    )
    emit_event(
        "tasks.completed",
        message="Text task bundle completed.",
        meta=result.meta,
        payload={
            "classification": result.classification.label,
            "route": result.route.name,
            "entity_count": len(result.extraction.entities),
            "status": result.status,
        },
    )
    return result
