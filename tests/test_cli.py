import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from lc_templates.cli import main
from lc_templates.core.schemas import (
    AgentExecutionResult,
    ClassificationResult,
    ConfigWarning,
    ExtractionResult,
    HealthReport,
    HealthSummary,
    KnowledgeBaseBuildResult,
    RouteDecision,
    TaskBundleResult,
    WorkflowExecutionResult,
)


class _App:
    def __init__(self):
        runtime_cls = type("Runtime", (), {"default_output_mode": "concise"})
        self._settings = type(
            "Settings",
            (),
            {
                "runtime": runtime_cls(),
                "model_dump": lambda self: {
                    "runtime": {"active_provider": "ollama", "default_output_mode": "concise"}
                }
            },
        )()

    def chat(self, text):
        return f"chat:{text}"

    def summarize(self, text):
        return f"summary:{text}"

    def classify(self, text, labels):
        return ClassificationResult(label=labels[0], reason=text, confidence=0.9)

    def route(self, text):
        return f"route:{text}"

    def route_decision(self, text):
        return RouteDecision(route=f"route:{text}", name=text, confidence=0.9)

    def route_name(self, text):
        return self.route(text).split(":", 1)[1]

    def route_display(self, text, verbose=False):
        if verbose:
            return f"Route: {self.route(text)}"
        return self.route_name(text)

    def extract(self, text):
        return ExtractionResult(entities=["A"], summary=f"extract:{text}")

    def extract_display(self, text, verbose=False):
        return f"Summary: extract:{text}" if verbose else f"extract:{text}"

    def agent(self, text):
        return AgentExecutionResult(final_text=text, used_tools=[], tool_call_count=0, raw={})

    def agent_display(self, text, verbose=False):
        return f"verbose:{text}" if verbose else text

    def memory_agent(self, thread_id, text):
        return AgentExecutionResult(
            final_text=f"{thread_id}:{text}", used_tools=[], tool_call_count=0, raw={}
        )

    def memory_agent_display(self, thread_id, text, verbose=False):
        base = f"{thread_id}:{text}"
        return f"verbose:{base}" if verbose else base

    def ask_rag_rendered(self, question, **_kwargs):
        return f"rag:{question}"

    def ask_rag_structured(self, question, **_kwargs):
        return type("Result", (), {"model_dump": lambda self: {"question": question}})()

    def index_file(self, path, **_kwargs):
        return KnowledgeBaseBuildResult(
            source_path=path,
            persist_directory="data/index/chroma",
            collection_name="demo",
            document_count=1,
            chunk_count=2,
        )

    def doctor(self):
        return HealthReport(
            package_version="0.1.0",
            active_provider="ollama",
            rerank_provider="ollama",
            config_path="config/config.yaml",
            default_collection_name="demo",
            default_persist_directory="data/index/chroma",
            ready=True,
            warnings=[ConfigWarning(code="demo", message="ok", severity="info")],
            recommendations=["Keep using concise mode for everyday runs."],
            recommended_middleware_profile="balanced",
            recommended_middleware_reason="Local development works well with balanced defaults.",
            summary=HealthSummary(
                ready=True,
                provider_count=1,
                ready_provider_count=1,
                warning_count=1,
            ),
            providers=[],
        )

    def doctor_display(self, verbose=False):
        return "Ready." if not verbose else "Ready: True"

    @property
    def settings(self):
        return self._settings

    def config(self):
        return self.settings.model_dump()

    def init_config(self, destination=None, overwrite=False):
        path = destination or "config/config.yaml"
        return path

    def run(self, text, **_kwargs):
        return WorkflowExecutionResult(
            route=RouteDecision(route="route:summarize", name="summarize", confidence=0.9),
            text=f"run:{text}",
            payload={"text": f"run:{text}"},
        )

    def run_display(self, text, verbose=False, **_kwargs):
        return f"Route: route:summarize\n\nResult:\nrun:{text}" if verbose else f"run:{text}"

    def index_display(self, path, **_kwargs):
        verbose = _kwargs.get("verbose", False)
        return f"Source: {path}" if verbose else "Indexed 1 document(s) into demo with 2 chunk(s)."

    def classify_text_result(self, text, labels, verbose=False):
        result = self.classify(text, labels)
        if verbose:
            return "\n".join(
                [
                    f"Label: {result.label}",
                    f"Confidence: {result.confidence:.2f}",
                    f"Reason: {result.reason}",
                ]
            )
        return f"{result.label} ({result.confidence:.2f})"

    def run_text_tasks(self, text):
        return TaskBundleResult(
            summary=f"summary:{text}",
            route=RouteDecision(route="route:chat", name="chat", confidence=0.9),
            classification=ClassificationResult(label="chat", reason="ok", confidence=0.9),
            extraction=ExtractionResult(summary=f"extract:{text}"),
        )

    def run_text_tasks_display(self, text, verbose=False):
        return f"summary:{text}" if not verbose else f"Route hint: chat\nextract:{text}"


class CliTests(unittest.TestCase):
    def test_chat_command(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.create_app", return_value=_App()):
            with redirect_stdout(buffer):
                exit_code = main(["chat", "hello"])
        self.assertEqual(exit_code, 0)
        self.assertIn("chat:hello", buffer.getvalue())

    def test_agent_command(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.create_app", return_value=_App()):
            with redirect_stdout(buffer):
                exit_code = main(["agent", "do work"])
        self.assertEqual(exit_code, 0)
        self.assertIn("do work", buffer.getvalue())

    def test_agent_command_json_output(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.create_app", return_value=_App()):
            with redirect_stdout(buffer):
                exit_code = main(["agent", "do work", "--output", "json"])
        self.assertEqual(exit_code, 0)
        self.assertIn('"final_text": "do work"', buffer.getvalue())

    def test_index_command(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.create_app", return_value=_App()):
            with redirect_stdout(buffer):
                exit_code = main(["index", "file.txt"])
        self.assertEqual(exit_code, 0)
        self.assertIn("Indexed 1 document(s)", buffer.getvalue())

    def test_index_command_json_output(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.create_app", return_value=_App()):
            with redirect_stdout(buffer):
                exit_code = main(["index", "file.txt", "--output", "json"])
        self.assertEqual(exit_code, 0)
        self.assertIn('"source_path": "file.txt"', buffer.getvalue())

    def test_doctor_command(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.create_app", return_value=_App()):
            with redirect_stdout(buffer):
                exit_code = main(["doctor"])
        self.assertEqual(exit_code, 0)
        self.assertIn("Ready.", buffer.getvalue())

    def test_version_command(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = main(["version"])
        self.assertEqual(exit_code, 0)
        self.assertTrue(buffer.getvalue().strip())

    def test_config_command(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.create_app", return_value=_App()):
            with redirect_stdout(buffer):
                exit_code = main(["config"])
        self.assertEqual(exit_code, 0)
        self.assertIn("active_provider: ollama", buffer.getvalue())

    def test_classify_command_concise_output(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.create_app", return_value=_App()):
            with redirect_stdout(buffer):
                exit_code = main(["classify", "hello", "--labels", "chat"])
        self.assertEqual(exit_code, 0)
        self.assertIn("chat (0.90)", buffer.getvalue())

    def test_extract_command_json_output(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.create_app", return_value=_App()):
            with redirect_stdout(buffer):
                exit_code = main(["extract", "hello", "--output", "json"])
        self.assertEqual(exit_code, 0)
        self.assertIn('"summary": "extract:hello"', buffer.getvalue())

    def test_tasks_command_concise_output(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.create_app", return_value=_App()):
            with redirect_stdout(buffer):
                exit_code = main(["tasks", "hello"])
        self.assertEqual(exit_code, 0)
        self.assertIn("summary:hello", buffer.getvalue())

    def test_route_command_json_output(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.create_app", return_value=_App()):
            with redirect_stdout(buffer):
                exit_code = main(["route", "chat", "--output", "json"])
        self.assertEqual(exit_code, 0)
        self.assertIn('"route": "route:chat"', buffer.getvalue())

    def test_run_command_json_output(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.create_app", return_value=_App()):
            with redirect_stdout(buffer):
                exit_code = main(["run", "hello", "--output", "json"])
        self.assertEqual(exit_code, 0)
        self.assertIn('"text": "run:hello"', buffer.getvalue())

    def test_run_command_accepts_rag_options(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.create_app", return_value=_App()):
            with redirect_stdout(buffer):
                exit_code = main(
                    ["run", "hello", "--use-rag", "--collection-name", "demo", "--output", "json"]
                )
        self.assertEqual(exit_code, 0)
        self.assertIn('"text": "run:hello"', buffer.getvalue())

    def test_init_config_command_json_output(self):
        buffer = io.StringIO()
        with patch("lc_templates.cli.scaffold_config", return_value="demo.yaml"):
            with redirect_stdout(buffer):
                exit_code = main(["init-config", "--path", "demo.yaml", "--output", "json"])
        self.assertEqual(exit_code, 0)
        self.assertIn('"path": "demo.yaml"', buffer.getvalue())
