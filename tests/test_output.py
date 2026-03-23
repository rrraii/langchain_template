
import unittest

from lc_templates.core.output import (
    build_agent_execution_result,
    coerce_response_text,
    parse_json_model,
    render_agent_execution_result,
    render_classification_result,
    render_extraction_result,
    render_route_result,
    render_task_result,
)
from lc_templates.core.schemas import ClassificationResult, ExtractionResult


class _Message:
    def __init__(self, content="", name="", tool_calls=None):
        self.content = content
        self.name = name
        self.tool_calls = tool_calls or []


class OutputTests(unittest.TestCase):
    def test_coerce_response_text_handles_list_content(self):
        message = _Message(
            content=[{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]
        )
        self.assertEqual(coerce_response_text(message), "hello world")

    def test_build_agent_execution_result_prefers_executed_tool_names(self):
        result = {
            "messages": [
                _Message(tool_calls=[{"name": "calculator"}]),
                _Message(content="42", name="calculator"),
                _Message(content="final answer"),
            ]
        }

        normalized = build_agent_execution_result(result)

        self.assertEqual(normalized.final_text, "final answer")
        self.assertEqual(normalized.used_tools, ["calculator"])
        self.assertEqual(normalized.tool_call_count, 1)
        self.assertEqual(normalized.tool_calls[0].name, "calculator")

    def test_parse_json_model_accepts_markdown_fenced_json(self):
        text = """```json
{"label":"chat","reason":"fallback","confidence":0.6}
```"""

        parsed = parse_json_model(text, ClassificationResult)

        self.assertEqual(parsed.label, "chat")
        self.assertEqual(parsed.confidence, 0.6)

    def test_render_classification_result_supports_concise_and_verbose(self):
        result = ClassificationResult(label="summarize", reason="clear intent", confidence=0.95)
        self.assertEqual(render_classification_result(result), "summarize (0.95)")
        self.assertIn("Reason: clear intent", render_classification_result(result, "verbose"))

    def test_render_agent_execution_result_supports_concise_and_verbose(self):
        result = build_agent_execution_result(
            {
                "messages": [
                    _Message(content="42", name="calculator"),
                    _Message(content="final answer"),
                ]
            }
        )
        self.assertIn("Tools: calculator", render_agent_execution_result(result))
        self.assertIn("Tool call count: 1", render_agent_execution_result(result, "verbose"))

    def test_render_extraction_result_supports_concise_and_verbose(self):
        result = ExtractionResult(entities=["发热", "咳嗽"], summary="疑似上呼吸道感染")
        self.assertIn("Entities: 发热, 咳嗽", render_extraction_result(result))
        self.assertIn("Entities:", render_extraction_result(result, "verbose"))

    def test_render_extraction_result_handles_empty_payload(self):
        result = ExtractionResult(entities=[], summary="")
        self.assertEqual(render_extraction_result(result), "No entities extracted.")

    def test_render_extraction_result_verbose_includes_status(self):
        result = ExtractionResult(entities=["A"], summary="B")
        self.assertIn("Status: ok", render_extraction_result(result, "verbose"))

    def test_render_route_result_strips_prefix_in_concise_mode(self):
        self.assertEqual(render_route_result("route:summarize"), "summarize")
        self.assertEqual(render_route_result("route:summarize", "verbose"), "route:summarize")

    def test_render_task_result_supports_concise_and_verbose(self):
        result = {
            "summary": "这是摘要",
            "classification": {"label": "medical"},
            "extraction": {"summary": "这是抽取摘要"},
        }
        concise = render_task_result(result)
        verbose = render_task_result(result, "verbose")
        self.assertIn("Route hint: medical", concise)
        self.assertIn('"summary": "这是摘要"', verbose)
