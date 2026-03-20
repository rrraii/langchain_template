from __future__ import annotations

import os
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from langchain_core.documents import Document

from lc_templates.agents.basic_agent import run_basic_agent
from lc_templates.agents.memory_agent import run_memory_agent
from lc_templates.chains.basic_chat import basic_chat
from lc_templates.chains.prompt_chain import summarize_text
from lc_templates.chains.streaming import batch_chat, stream_chat
from lc_templates.chains.structured_output import classify_text, extract_entities
from lc_templates.chains.tasks import run_text_tasks
from lc_templates.core.config import get_resolved_config_path, get_settings, resolve_runtime_path
from lc_templates.core.logging import configure_logging_from_runtime
from lc_templates.core.output import (
    build_agent_execution_result,
    render_agent_execution_result,
    render_classification_result,
    render_extraction_result,
    render_route_result,
    render_task_result,
)
from lc_templates.core.schemas import (
    AgentExecutionResult,
    ClassificationResult,
    ConfigWarning,
    ExtractionResult,
    GroundedAnswer,
    HealthReport,
    KnowledgeBaseBuildResult,
    ProviderStatus,
)
from lc_templates.rag.indexing import build_vector_store, load_vector_store, resolve_collection_name
from lc_templates.rag.loaders import load_documents
from lc_templates.rag.pipeline import answer_with_rag, answer_with_structured_rag, render_rag_answer
from lc_templates.rag.retrievers import create_vector_retriever
from lc_templates.rag.splitters import split_documents
from lc_templates.workflows.router import route_task


class TemplateApp:
    """High-level facade for the template library."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path
        if config_path:
            os.environ["LC_TEMPLATES_CONFIG"] = config_path
            get_settings.cache_clear()
            self.settings = get_settings(config_path)
        else:
            self.settings = get_settings()
        configure_logging_from_runtime(self.settings.runtime)

    def chat(self, text: str) -> str:
        return basic_chat(text)

    def version(self) -> str:
        try:
            return version("langchain12-templates")
        except PackageNotFoundError:
            return "0.1.0"

    def config(self) -> dict:
        return self.settings.model_dump()

    def stream_chat(self, text: str) -> Iterable[str]:
        return stream_chat(text)

    def batch_chat(self, prompts: list[str]) -> list[str]:
        return batch_chat(prompts)

    def summarize(self, text: str) -> str:
        return summarize_text(text)

    def classify(self, text: str, labels: list[str]) -> ClassificationResult:
        return classify_text(text, labels)

    def classify_label(self, text: str, labels: list[str]) -> str:
        return self.classify(text, labels).label

    def classify_text_result(self, text: str, labels: list[str], *, verbose: bool = False) -> str:
        return render_classification_result(
            self.classify(text, labels), detail_level="verbose" if verbose else "concise"
        )

    def extract(self, text: str) -> ExtractionResult:
        return extract_entities(text)

    def extract_text(self, text: str) -> str:
        return self.extract(text).summary

    def extract_display(self, text: str, *, verbose: bool = False) -> str:
        return render_extraction_result(
            self.extract(text), detail_level="verbose" if verbose else "concise"
        )

    def run_text_tasks(self, text: str) -> dict:
        return run_text_tasks(text)

    def run_text_tasks_display(self, text: str, *, verbose: bool = False) -> str:
        return render_task_result(
            self.run_text_tasks(text), detail_level="verbose" if verbose else "concise"
        )

    def route(self, text: str) -> str:
        return route_task(text)

    def route_name(self, text: str) -> str:
        return render_route_result(self.route(text))

    def route_display(self, text: str, *, verbose: bool = False) -> str:
        return render_route_result(
            self.route(text), detail_level="verbose" if verbose else "concise"
        )

    def agent(self, text: str) -> AgentExecutionResult:
        return build_agent_execution_result(run_basic_agent(text))

    def agent_text(self, text: str) -> str:
        return self.agent(text).final_text

    def agent_display(self, text: str, *, verbose: bool = False) -> str:
        return render_agent_execution_result(
            self.agent(text), detail_level="verbose" if verbose else "concise"
        )

    def memory_agent(self, thread_id: str, text: str) -> AgentExecutionResult:
        return build_agent_execution_result(run_memory_agent(thread_id, text))

    def memory_agent_text(self, thread_id: str, text: str) -> str:
        return self.memory_agent(thread_id, text).final_text

    def memory_agent_display(self, thread_id: str, text: str, *, verbose: bool = False) -> str:
        return render_agent_execution_result(
            self.memory_agent(thread_id, text),
            detail_level="verbose" if verbose else "concise",
        )

    def load_documents(self, path: str) -> list[Document]:
        return load_documents(path)

    def split_documents(
        self,
        documents: list[Document],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Document]:
        return split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def build_knowledge_base(
        self,
        path: str,
        *,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        provider_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        documents = self.load_documents(path)
        chunks = self.split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return build_vector_store(
            chunks,
            persist_directory=persist_directory,
            collection_name=collection_name,
            provider_name=provider_name,
        )

    def index_file(
        self,
        path: str,
        *,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        provider_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> KnowledgeBaseBuildResult:
        documents = self.load_documents(path)
        chunks = self.split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        resolved_provider_name = provider_name or self.settings.get_active_provider_name()
        provider = self.settings.get_provider(resolved_provider_name)
        resolved_collection_name = resolve_collection_name(
            collection_name,
            provider_name=resolved_provider_name,
            settings=self.settings,
        )
        build_vector_store(
            chunks,
            persist_directory=persist_directory,
            collection_name=resolved_collection_name,
            provider_name=resolved_provider_name,
        )
        return KnowledgeBaseBuildResult(
            source_path=path,
            persist_directory=str(
                resolve_runtime_path(
                    persist_directory or self.settings.runtime.default_persist_directory
                )
            ),
            collection_name=resolved_collection_name,
            provider_name=resolved_provider_name,
            embedding_model=provider.embedding_model,
            embedding_dimensions=provider.embedding_dimensions,
            document_count=len(documents),
            chunk_count=len(chunks),
        )

    def get_retriever(
        self,
        *,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        provider_name: str | None = None,
        k: int | None = None,
    ):
        store = load_vector_store(
            persist_directory=persist_directory,
            collection_name=collection_name,
            provider_name=provider_name,
        )
        return create_vector_retriever(store, k=k or self.settings.runtime.top_k)

    def ask_rag(
        self,
        question: str,
        *,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        provider_name: str | None = None,
        k: int | None = None,
    ) -> str:
        retriever = self.get_retriever(
            persist_directory=persist_directory,
            collection_name=collection_name,
            provider_name=provider_name,
            k=k,
        )
        return answer_with_rag(retriever, question)

    def ask_rag_structured(
        self,
        question: str,
        *,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        provider_name: str | None = None,
        k: int | None = None,
    ) -> GroundedAnswer:
        retriever = self.get_retriever(
            persist_directory=persist_directory,
            collection_name=collection_name,
            provider_name=provider_name,
            k=k,
        )
        return answer_with_structured_rag(retriever, question)

    def ask_rag_rendered(
        self,
        question: str,
        *,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        provider_name: str | None = None,
        k: int | None = None,
    ) -> str:
        retriever = self.get_retriever(
            persist_directory=persist_directory,
            collection_name=collection_name,
            provider_name=provider_name,
            k=k,
        )
        return render_rag_answer(retriever, question)

    def ask_rag_from_file(
        self,
        path: str,
        question: str,
        *,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        provider_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        k: int | None = None,
    ) -> str:
        self.index_file(
            path,
            persist_directory=persist_directory,
            collection_name=collection_name,
            provider_name=provider_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return self.ask_rag_rendered(
            question,
            persist_directory=persist_directory,
            collection_name=collection_name,
            provider_name=provider_name,
            k=k,
        )

    def doctor(self) -> HealthReport:
        resolved_path = get_resolved_config_path(self.config_path)
        providers: list[ProviderStatus] = []
        warnings: list[ConfigWarning] = []
        provider_dump = self.settings.providers.model_dump()
        active_provider_name = self.settings.get_active_provider_name()
        rerank_provider_name = self.settings.get_rerank_provider_name()

        if not Path(resolved_path).exists():
            warnings.append(
                ConfigWarning(
                    code="config_missing",
                    message=f"Config file does not exist: {resolved_path}",
                    severity="error",
                )
            )
        if self.settings.runtime.chunk_overlap >= self.settings.runtime.chunk_size:
            warnings.append(
                ConfigWarning(
                    code="chunk_overlap_invalid",
                    message="chunk_overlap should be smaller than chunk_size for stable splitting.",
                )
            )

        for name in provider_dump:
            provider = getattr(self.settings.providers, name)
            provider_warnings: list[str] = []
            has_api_key = provider.type == "ollama" or not provider.has_placeholder_api_key()
            ready = provider.enabled
            if not provider.enabled:
                provider_warnings.append("Provider is disabled.")
                ready = False
            if provider.type == "openai_compatible" and not has_api_key:
                provider_warnings.append("API key is missing or still set to a placeholder value.")
                ready = False
            if provider.type == "openai_compatible" and not provider.base_url:
                provider_warnings.append("base_url is required for openai_compatible providers.")
                ready = False
            if provider.type == "ollama" and not provider.chat_model:
                provider_warnings.append("chat_model is required for ollama providers.")
                ready = False
            if not provider.embedding_model:
                provider_warnings.append(
                    "embedding_model is not configured. RAG indexing and retrieval will not work."
                )
            elif provider.embedding_dimensions is None:
                provider_warnings.append(
                    "embedding_dimensions is not configured. "
                    "Default collection isolation will be less explicit."
                )

            if name == active_provider_name and not ready:
                joined_warnings = " ".join(provider_warnings)
                warnings.append(
                    ConfigWarning(
                        code="active_provider_not_ready",
                        message=f"Active provider '{name}' is not ready: {joined_warnings}",
                        severity="error",
                    )
                )
            if (
                name == rerank_provider_name
                and provider.type == "openai_compatible"
                and not has_api_key
            ):
                warnings.append(
                    ConfigWarning(
                        code="rerank_provider_not_ready",
                        message=f"Rerank provider '{name}' is not ready for remote calls.",
                    )
                )

            providers.append(
                ProviderStatus(
                    name=name,
                    enabled=provider.enabled,
                    provider_type=provider.type,
                    has_api_key=has_api_key,
                    base_url=provider.base_url,
                    chat_model=provider.chat_model,
                    embedding_model=provider.embedding_model,
                    embedding_dimensions=provider.embedding_dimensions,
                    ready=ready,
                    warnings=provider_warnings,
                )
            )

        return HealthReport(
            package_version=self.version(),
            active_provider=active_provider_name,
            rerank_provider=rerank_provider_name,
            config_path=str(Path(resolved_path)),
            default_collection_name=self.settings.runtime.default_collection_name,
            default_persist_directory=str(
                resolve_runtime_path(self.settings.runtime.default_persist_directory)
            ),
            ready=not any(item.severity == "error" for item in warnings),
            warnings=warnings,
            providers=providers,
        )


def create_app(config_path: str | None = None) -> TemplateApp:
    return TemplateApp(config_path=config_path)
