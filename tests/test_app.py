import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lc_templates.app import TemplateApp
from lc_templates.core.hooks import clear_event_hooks
from lc_templates.core.schemas import (
    AgentExecutionResult,
    ClassificationResult,
    ExtractionResult,
    GroundedAnswer,
    HealthReport,
    KnowledgeBaseBuildResult,
    RouteDecision,
    TaskBundleResult,
    WorkflowExecutionResult,
)


class AppFacadeTests(unittest.TestCase):
    def setUp(self):
        clear_event_hooks()

    def test_chat_delegates_to_basic_chat(self):
        with patch("lc_templates.app.basic_chat", return_value="ok") as mocked:
            app = TemplateApp()
            self.assertEqual(app.chat("hello"), "ok")
            mocked.assert_called_once_with("hello")

    def test_version_returns_string(self):
        app = TemplateApp()
        self.assertTrue(app.version())

    def test_config_returns_dict(self):
        app = TemplateApp()
        result = app.config()
        self.assertIn("runtime", result)
        self.assertIn("providers", result)

    def test_classify_label_returns_label_only(self):
        classification = ClassificationResult(label="chat", reason="fallback", confidence=0.9)
        with patch("lc_templates.app.classify_text", return_value=classification):
            app = TemplateApp()
            result = app.classify_label("text", ["chat"])
        self.assertEqual(result, "chat")

    def test_classify_adds_trace_and_latency(self):
        classification = ClassificationResult(label="chat", reason="fallback", confidence=0.9)
        with patch("lc_templates.app.classify_text", return_value=classification):
            app = TemplateApp()
            result = app.classify("text", ["chat"])
        self.assertTrue(result.trace_id)
        self.assertGreaterEqual(result.latency_ms, 0.0)

    def test_agent_text_returns_final_text_only(self):
        normalized = AgentExecutionResult(
            final_text="done", used_tools=["calculator"], tool_call_count=1, raw={}
        )
        with patch("lc_templates.app.run_basic_agent", return_value={"messages": []}):
            with patch("lc_templates.app.build_agent_execution_result", return_value=normalized):
                app = TemplateApp()
                result = app.agent_text("hello")
        self.assertEqual(result, "done")

    def test_extract_text_returns_summary_only(self):
        extraction = ExtractionResult(entities=["A"], summary="summary")
        with patch("lc_templates.app.extract_entities", return_value=extraction):
            app = TemplateApp()
            result = app.extract_text("text")
        self.assertEqual(result, "summary")

    def test_route_name_returns_normalized_name(self):
        decision = RouteDecision(route="route:summarize", name="summarize", confidence=0.9)
        with patch("lc_templates.app.route_decision", return_value=decision):
            app = TemplateApp()
            result = app.route_name("text")
        self.assertEqual(result, "summarize")

    def test_run_text_tasks_display_renders_text(self):
        payload = TaskBundleResult(
            summary="demo summary",
            route=RouteDecision(route="route:medical", name="medical", confidence=0.9),
            classification=ClassificationResult(label="medical", reason="clear", confidence=0.9),
            extraction=ExtractionResult(summary="demo extraction"),
        )
        with patch("lc_templates.app.run_text_tasks", return_value=payload):
            app = TemplateApp()
            result = app.run_text_tasks_display("text")
        self.assertIn("Route hint: medical", result)

    def test_run_returns_workflow_result(self):
        decision = RouteDecision(route="route:summarize", name="summarize", confidence=0.9)
        with patch("lc_templates.app.route_decision", return_value=decision):
            with patch("lc_templates.app.summarize_text", return_value="short summary"):
                app = TemplateApp()
                result = app.run("text")
        self.assertIsInstance(result, WorkflowExecutionResult)
        self.assertEqual(result.text, "short summary")
        self.assertEqual(result.route.name, "summarize")

    def test_run_uses_rag_when_enabled(self):
        decision = RouteDecision(route="route:rag", name="rag", confidence=0.9)
        grounded = GroundedAnswer(answer="rag answer", citations=["[1]"], grounded=True)
        with patch("lc_templates.app.route_decision", return_value=decision):
            with patch.object(TemplateApp, "ask_rag_structured", return_value=grounded):
                app = TemplateApp()
                result = app.run("question", use_rag=True)
        self.assertEqual(result.text, "rag answer\n\nReferences\n- [1]")
        self.assertEqual(result.route.name, "rag")

    def test_agent_returns_normalized_result(self):
        normalized = AgentExecutionResult(
            final_text="done", used_tools=["calculator"], tool_call_count=1, raw={}
        )
        with patch("lc_templates.app.run_basic_agent", return_value={"messages": []}) as mocked_run:
            with patch(
                "lc_templates.app.build_agent_execution_result", return_value=normalized
            ) as mocked_build:
                app = TemplateApp()
                result = app.agent("hello")

        self.assertEqual(result.final_text, "done")
        mocked_run.assert_called_once_with("hello")
        mocked_build.assert_called_once()

    def test_rag_structured_delegates_to_pipeline(self):
        grounded = GroundedAnswer(answer="ok", citations=["[1]"], grounded=True)
        with patch("lc_templates.app.load_vector_store", return_value=object()):
            with patch(
                "lc_templates.app.create_vector_retriever", return_value="retriever"
            ) as mocked_retriever:
                with patch(
                    "lc_templates.app.answer_with_structured_rag", return_value=grounded
                ) as mocked_answer:
                    app = TemplateApp()
                    result = app.ask_rag_structured("question")

        self.assertTrue(result.grounded)
        mocked_retriever.assert_called_once()
        mocked_answer.assert_called_once_with("retriever", "question")

    def test_classify_delegates_to_chain(self):
        classification = ClassificationResult(label="chat", reason="fallback", confidence=0.9)
        with patch("lc_templates.app.classify_text", return_value=classification) as mocked:
            app = TemplateApp()
            result = app.classify("text", ["chat"])

        self.assertEqual(result.label, "chat")
        mocked.assert_called_once_with("text", ["chat"])

    def test_extract_delegates_to_chain(self):
        extraction = ExtractionResult(entities=["A"], summary="B")
        with patch("lc_templates.app.extract_entities", return_value=extraction) as mocked:
            app = TemplateApp()
            result = app.extract("text")

        self.assertEqual(result.entities, ["A"])
        mocked.assert_called_once_with("text")

    def test_index_file_returns_metadata(self):
        with patch("lc_templates.app.load_documents", return_value=[object(), object()]):
            with patch("lc_templates.app.split_documents", return_value=[1, 2, 3]) as mocked_split:
                with patch("lc_templates.app.build_vector_store") as mocked_store:
                    app = TemplateApp()
                    result = app.index_file("demo.txt")

        self.assertIsInstance(result, KnowledgeBaseBuildResult)
        self.assertEqual(result.document_count, 2)
        self.assertEqual(result.chunk_count, 3)
        self.assertTrue(result.collection_name.startswith("demo_collection__"))
        self.assertEqual(result.provider_name, app.settings.get_active_provider_name())
        self.assertTrue(result.embedding_model)
        mocked_split.assert_called_once()
        mocked_store.assert_called_once()

    def test_doctor_returns_health_report(self):
        app = TemplateApp()
        result = app.doctor()

        self.assertIsInstance(result, HealthReport)
        self.assertEqual(result.active_provider, app.settings.get_active_provider_name())
        self.assertTrue(result.package_version)
        self.assertGreaterEqual(len(result.providers), 1)
        self.assertEqual(
            Path(result.default_persist_directory).parts[-3:],
            ("data", "index", "chroma"),
        )
        self.assertEqual(result.summary.provider_count, len(result.providers))

    def test_doctor_summary_returns_aggregate_health(self):
        app = TemplateApp()
        result = app.doctor_summary()

        self.assertGreaterEqual(result.provider_count, 1)

    def test_doctor_recommendations_returns_list(self):
        app = TemplateApp()
        result = app.doctor_recommendations()
        self.assertIsInstance(result, list)

    def test_doctor_recommended_profile_returns_string(self):
        app = TemplateApp()
        result = app.doctor_recommended_profile()
        self.assertIn(result, {"safe", "balanced", "aggressive", "custom"})

    def test_doctor_display_returns_text(self):
        app = TemplateApp()
        result = app.doctor_display()
        self.assertTrue(result)

    def test_on_event_receives_completed_result_events(self):
        classification = ClassificationResult(label="chat", reason="fallback", confidence=0.9)
        received = []
        with patch("lc_templates.app.classify_text", return_value=classification):
            app = TemplateApp()
            app.on_event(received.append)
            app.classify("text", ["chat"])

        self.assertTrue(received)
        self.assertEqual(received[-1].name, "classify.completed")
        self.assertEqual(received[-1].meta.operation, "classify")
        self.assertTrue(received[-1].trace_id)

    def test_init_config_writes_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "config.yaml"
            app = TemplateApp()
            result = app.init_config(str(target))
            self.assertEqual(result, str(target))
            self.assertTrue(target.exists())
