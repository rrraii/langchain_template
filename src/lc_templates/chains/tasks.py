from lc_templates.chains.prompt_chain import summarize_text
from lc_templates.chains.structured_output import classify_text, extract_entities
from lc_templates.core.logging import get_logger

TASK_LABELS = ["medical", "legal", "customer_service", "general"]
logger = get_logger(__name__)


def run_text_tasks(text: str) -> dict:
    logger.info("Running text task bundle.")
    result = {
        "summary": summarize_text(text),
        "classification": classify_text(text, TASK_LABELS).model_dump(),
        "extraction": extract_entities(text).model_dump(),
    }
    logger.info(
        "Text task bundle completed. classification=%s extraction_entities=%s",
        result["classification"].get("label", ""),
        len(result["extraction"].get("entities", [])),
    )
    return result
