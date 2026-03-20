from __future__ import annotations

import json

from pydantic import BaseModel

from lc_templates.core.logging import get_logger
from lc_templates.core.models import build_chat_model
from lc_templates.core.output import coerce_response_text, parse_json_model, strip_json_fences
from lc_templates.core.prompts import build_classification_prompt, build_extraction_prompt
from lc_templates.core.schemas import ClassificationResult, ExtractionResult

logger = get_logger(__name__)


def _build_json_required_messages(prompt_value) -> list:
    if hasattr(prompt_value, "to_messages"):
        messages = list(prompt_value.to_messages())
    elif isinstance(prompt_value, list):
        messages = list(prompt_value)
    else:
        messages = [("human", str(prompt_value))]

    return [
        (
            "system",
            "Return a valid JSON object. "
            "The response must be JSON and must match the required schema.",
        ),
        *messages,
    ]


def _invoke_structured_with_fallback(prompt_value, schema: type[BaseModel]) -> BaseModel:
    model = build_chat_model()
    structured_messages = _build_json_required_messages(prompt_value)
    try:
        logger.debug("Structured output request started for schema=%s", schema.__name__)
        return model.with_structured_output(schema).invoke(structured_messages)
    except Exception as exc:
        logger.warning(
            "Structured output request failed for schema=%s; switching to JSON fallback. error=%s",
            schema.__name__,
            exc,
        )
        fallback_prompt = _build_json_required_messages(prompt_value)
        raw_response = model.invoke(fallback_prompt)
        logger.debug("JSON fallback response received for schema=%s", schema.__name__)
        return parse_json_model(coerce_response_text(raw_response), schema)


def classify_text(text: str, labels: list[str]) -> ClassificationResult:
    prompt = build_classification_prompt(labels)
    result = _invoke_structured_with_fallback(prompt.invoke({"text": text}), ClassificationResult)
    if result.label not in labels:
        normalized = result.label.strip().lower()
        label_map = {label.strip().lower(): label for label in labels}
        result.label = label_map.get(normalized, labels[0])
    result.reason = result.reason.strip()
    result.confidence = min(1.0, max(0.0, float(result.confidence)))
    return result


def _normalize_extraction_payload(payload: object) -> ExtractionResult:
    if isinstance(payload, ExtractionResult):
        return payload

    data = dict(payload) if isinstance(payload, dict) else {}
    raw_entities = data.get("entities", [])
    normalized_entities: list[str] = []

    if isinstance(raw_entities, list):
        for item in raw_entities:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    normalized_entities.append(text)
                continue

            if isinstance(item, dict):
                value = str(item.get("value", "")).strip()
                entity_type = str(item.get("type", "")).strip()
                if value and entity_type:
                    normalized_entities.append(f"{entity_type}: {value}")
                elif value:
                    normalized_entities.append(value)
                elif entity_type:
                    normalized_entities.append(entity_type)
                continue

            text = str(item).strip()
            if text:
                normalized_entities.append(text)

    summary = str(data.get("summary", "")).strip()
    status = str(data.get("status", "ok")).strip() or "ok"
    fallback_used = bool(data.get("fallback_used", False))
    error_reason = str(data.get("error_reason", "")).strip()
    logger.debug(
        "Normalized extraction payload. entity_count=%s summary_present=%s",
        len(normalized_entities),
        bool(summary),
    )
    return ExtractionResult(
        entities=normalized_entities,
        summary=summary,
        status=status,
        fallback_used=fallback_used,
        error_reason=error_reason,
    )


def _unavailable_extraction_result() -> ExtractionResult:
    logger.error("Extraction failed after all fallback attempts; returning unavailable result.")
    return ExtractionResult(
        summary="Extraction temporarily unavailable.",
        entities=[],
        status="unavailable",
        fallback_used=True,
        error_reason="double_fallback_failure",
    )


def extract_entities(text: str) -> ExtractionResult:
    prompt = build_extraction_prompt()
    try:
        result = _invoke_structured_with_fallback(prompt.invoke({"text": text}), ExtractionResult)
    except Exception as exc:
        logger.warning(
            "Extraction schema pipeline failed; attempting tolerant JSON recovery. error=%s",
            exc,
        )
        try:
            model = build_chat_model()
            fallback_prompt = _build_json_required_messages(prompt.invoke({"text": text}))
            raw_response = model.invoke(fallback_prompt)
            payload = json.loads(strip_json_fences(coerce_response_text(raw_response)))
            result = _normalize_extraction_payload(payload)
            result.fallback_used = True
            if not result.error_reason:
                result.error_reason = "structured_output_failed"
            logger.info(
                "Extraction recovered via tolerant JSON normalization. entity_count=%s",
                len(result.entities),
            )
        except Exception as recovery_exc:
            logger.exception(
                "Extraction tolerant recovery failed; returning unavailable result. error=%s",
                recovery_exc,
            )
            result = _unavailable_extraction_result()

    result = _normalize_extraction_payload(result.model_dump())
    result.entities = [entity.strip() for entity in result.entities if entity.strip()]
    result.summary = result.summary.strip()
    logger.info(
        "Extraction completed. status=%s fallback_used=%s entity_count=%s summary_present=%s",
        result.status,
        result.fallback_used,
        len(result.entities),
        bool(result.summary),
    )
    return result
