import unittest
from unittest.mock import Mock, patch

from lc_templates.chains.structured_output import (
    _build_json_required_messages,
    _normalize_extraction_payload,
    _unavailable_extraction_result,
    extract_entities,
)
from lc_templates.core.prompts import build_classification_prompt


class StructuredOutputTests(unittest.TestCase):
    def test_build_json_required_messages_adds_json_instruction(self):
        messages = _build_json_required_messages([("human", "hello")])

        self.assertIn("JSON", messages[0][1])

    def test_classification_prompt_mentions_json(self):
        prompt_value = build_classification_prompt(["medical", "general"]).invoke({"text": "demo"})
        joined = "\n".join(str(message.content) for message in prompt_value.to_messages())

        self.assertIn("JSON", joined)

    def test_normalize_extraction_payload_accepts_entity_objects(self):
        result = _normalize_extraction_payload(
            {
                "entities": [
                    {"type": "症状", "value": "头晕"},
                    {"type": "生命体征", "value": "血压 160/100 mmHg"},
                ],
                "summary": "患者近一周头晕。",
            }
        )

        self.assertEqual(
            result.entities,
            ["症状: 头晕", "生命体征: 血压 160/100 mmHg"],
        )
        self.assertEqual(result.summary, "患者近一周头晕。")

    def test_unavailable_extraction_result_is_non_fatal(self):
        result = _unavailable_extraction_result()

        self.assertEqual(result.summary, "Extraction temporarily unavailable.")
        self.assertEqual(result.entities, [])
        self.assertEqual(result.status, "unavailable")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.error_reason, "double_fallback_failure")

    def test_extract_entities_returns_fallback_result_on_double_failure(self):
        fake_prompt = Mock()
        prompt_value = Mock()
        prompt_value.to_messages.return_value = [("human", "text")]
        fake_prompt.invoke.return_value = prompt_value

        fake_model = Mock()
        fake_model.invoke.side_effect = RuntimeError("timeout")

        with patch(
            "lc_templates.chains.structured_output.build_extraction_prompt",
            return_value=fake_prompt,
        ):
            with patch(
                "lc_templates.chains.structured_output._invoke_structured_with_fallback",
                side_effect=RuntimeError("schema failure"),
            ):
                with patch(
                    "lc_templates.chains.structured_output.build_chat_model",
                    return_value=fake_model,
                ):
                    result = extract_entities("demo")

        self.assertEqual(result.summary, "Extraction temporarily unavailable.")
        self.assertEqual(result.entities, [])
        self.assertEqual(result.status, "unavailable")
