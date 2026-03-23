from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from lc_templates.core.config import get_settings
from lc_templates.core.models import build_chat_model
from lc_templates.core.output import (
    coerce_response_text,
    normalize_text,
    parse_json_model,
    render_grounded_answer,
)
from lc_templates.core.prompts import build_qa_prompt
from lc_templates.core.schemas import CitationItem, ExecutionMetadata, GroundedAnswer


def format_docs(documents: list[Document]) -> str:
    formatted_chunks: list[str] = []
    for index, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", f"doc-{index}")
        formatted_chunks.append(f"[{index}] source={source}\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)


def _trim_citations(citations: list[str]) -> list[str]:
    max_citations = get_settings().runtime.max_citations
    return [normalize_text(item) for item in citations if normalize_text(item)][:max_citations]


def _build_citation_candidates(documents: list[Document]) -> list[str]:
    candidates: list[str] = []
    for index, doc in enumerate(documents, 1):
        source = str(doc.metadata.get("source", f"doc-{index}")).strip()
        candidates.append(f"[{index}]")
        candidates.append(f"source={source}")

        content = normalize_text(doc.page_content)
        if content:
            candidates.append(content)
            snippet = content[:120].strip()
            if snippet:
                candidates.append(snippet)
    return candidates


def _validate_citations(citations: list[str], documents: list[Document]) -> list[str]:
    normalized_candidates = [
        candidate.lower() for candidate in _build_citation_candidates(documents) if candidate
    ]
    validated: list[str] = []
    for citation in _trim_citations(citations):
        lowered = citation.lower()
        if any(lowered in candidate or candidate in lowered for candidate in normalized_candidates):
            validated.append(citation)
    return validated


def _build_citation_items(citations: list[str], documents: list[Document]) -> list[CitationItem]:
    items: list[CitationItem] = []
    validated = _validate_citations(citations, documents)
    for citation in validated:
        marker = ""
        source = ""
        for index, doc in enumerate(documents, 1):
            doc_source = str(doc.metadata.get("source", f"doc-{index}")).strip()
            if f"[{index}]" in citation:
                marker = f"[{index}]"
            if doc_source and doc_source in citation:
                source = doc_source
            if marker or source:
                break
        items.append(CitationItem(marker=marker, source=source, snippet=citation))
    return items


def _build_grounded_prompt(context: str, question: str) -> list[tuple[str, str]]:
    settings = get_settings().runtime
    return [
        (
            "system",
            "\n".join(
                [
                    "You are a retrieval-grounded QA assistant.",
                    f"Respond in {settings.response_language}.",
                    "Return valid JSON only without Markdown fences.",
                    "The JSON must match this schema: "
                    '{"answer": str, "citations": list[str], "grounded": bool}.',
                    "Use grounded=false when the context is insufficient.",
                    f"If grounded is false, set answer to: {settings.rag_no_answer_message}",
                ]
            ),
        ),
        (
            "human",
            f"Context:\n{context}\n\nQuestion:\n{question}\n\n"
            "Citations should be short snippets or source markers copied from the context.",
        ),
    ]


def _default_rag_meta() -> ExecutionMetadata:
    settings = get_settings()
    provider_name = settings.get_active_provider_name()
    provider = settings.get_active_provider()
    return ExecutionMetadata(
        provider_name=provider_name,
        model_name=provider.chat_model,
        operation="rag",
    )


def build_rag_chain(retriever):
    prompt = build_qa_prompt()
    model = build_chat_model()
    return (
        {
            "context": RunnableLambda(retriever.invoke) | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
        | normalize_text
    )


def answer_with_rag(retriever, question: str) -> str:
    return build_rag_chain(retriever).invoke(question)


def answer_with_structured_rag(retriever, question: str) -> GroundedAnswer:
    model = build_chat_model()
    documents = retriever.invoke(question)
    context = format_docs(documents)
    prompt_messages = _build_grounded_prompt(context, question)

    try:
        result = model.with_structured_output(GroundedAnswer).invoke(prompt_messages)
    except Exception:
        raw_response = model.invoke(prompt_messages)
        result = parse_json_model(coerce_response_text(raw_response), GroundedAnswer)

    result.meta = _default_rag_meta()
    result.answer = normalize_text(result.answer)
    result.citations = _validate_citations(result.citations, documents)
    result.citation_items = _build_citation_items(result.citations, documents)
    if not context.strip():
        result.grounded = False
        result.status = "unavailable"
        result.fallback_used = True
        result.error_reason = "empty_context"
        result.answer = get_settings().runtime.rag_no_answer_message
        result.citations = []
        result.citation_items = []
    elif result.grounded and not result.citations:
        result.grounded = False
        result.status = "partial"
        result.fallback_used = True
        result.error_reason = "missing_valid_citations"
        result.answer = get_settings().runtime.rag_no_answer_message
    return result


def render_rag_answer(retriever, question: str) -> str:
    return render_grounded_answer(answer_with_structured_rag(retriever, question))
