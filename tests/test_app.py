
import unittest
from pathlib import Path
from unittest.mock import patch

from lc_templates.app import TemplateApp
from lc_templates.core.schemas import (
    AgentExecutionResult,
    ClassificationResult,
    ExtractionResult,
    GroundedAnswer,
    HealthReport,
    KnowledgeBaseBuildResult,
)


class AppFacadeTests(unittest.TestCase):
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
        with patch("lc_templates.app.route_task", return_value="route:summarize"):
            app = TemplateApp()
            result = app.route_name("text")
        self.assertEqual(result, "summarize")

    def test_run_text_tasks_display_renders_text(self):
        payload = {
            "summary": "这是摘要",
            "classification": {"label": "medical"},
            "extraction": {"summary": "这是抽取摘要"},
        }
        with patch("lc_templates.app.run_text_tasks", return_value=payload):
            app = TemplateApp()
            result = app.run_text_tasks_display("text")
        self.assertIn("Route hint: medical", result)

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
