import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from lc_templates.core.config import get_settings
from lc_templates.rag import pipeline


class _Retriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, _question):
        return self.docs


class _ModelResponse:
    def __init__(self, content):
        self.content = content


class _FailStructuredModel:
    def with_structured_output(self, _schema):
        raise RuntimeError("structured output unavailable")

    def invoke(self, _messages):
        return _ModelResponse(
            '{"answer":"Please use a low-salt diet.",'
            '"citations":["[1]","hallucinated source"],"grounded":true}'
        )


class _HallucinatedCitationModel:
    def with_structured_output(self, _schema):
        raise RuntimeError("structured output unavailable")

    def invoke(self, _messages):
        return _ModelResponse(
            '{"answer":"Please use a low-salt diet.",'
            '"citations":["hallucinated source"],"grounded":true}'
        )


class RagPipelineTests(unittest.TestCase):
    def setUp(self):
        get_settings.cache_clear()

    def test_validate_citations_filters_out_unsupported_references(self):
        docs = [
            Document(
                page_content="Patients with hypertension should use a low-salt diet.",
                metadata={"source": "doc-a.txt"},
            )
        ]
        citations = pipeline._validate_citations(["[1]", "source=doc-a.txt", "fake"], docs)
        self.assertEqual(citations, ["[1]", "source=doc-a.txt"])

    def test_answer_with_structured_rag_marks_ungrounded_when_no_valid_citations(self):
        docs = [
            Document(
                page_content="Patients with hypertension should use a low-salt diet.",
                metadata={"source": "doc-a.txt"},
            )
        ]
        retriever = _Retriever(docs)

        with patch.object(
            pipeline,
            "build_chat_model",
            return_value=_HallucinatedCitationModel(),
        ):
            result = pipeline.answer_with_structured_rag(
                retriever,
                "What should patients with hypertension pay attention to?",
            )

        self.assertFalse(result.grounded)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.answer, get_settings().runtime.rag_no_answer_message)

    def test_answer_with_structured_rag_handles_empty_context(self):
        retriever = _Retriever([])

        with patch.object(pipeline, "build_chat_model", return_value=_FailStructuredModel()):
            result = pipeline.answer_with_structured_rag(
                retriever,
                "What should patients with hypertension pay attention to?",
            )

        self.assertFalse(result.grounded)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.answer, get_settings().runtime.rag_no_answer_message)
