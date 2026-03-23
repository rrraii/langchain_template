from __future__ import annotations

import os
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from langchain_core.documents import Document

from lc_templates.agents.basic_agent import run_basic_agent
from lc_templates.agents.memory_agent import run_memory_agent
from lc_templates.chains.basic_chat import basic_chat
from lc_templates.chains.prompt_chain import summarize_text
from lc_templates.chains.streaming import batch_chat, stream_chat
from lc_templates.chains.structured_output import classify_text, extract_entities
from lc_templates.chains.tasks import run_text_tasks
from lc_templates.core.checkpoint import build_memory_checkpointer
from lc_templates.core.config import (
    get_resolved_config_path,
    get_settings,
    resolve_runtime_path,
    scaffold_config,
)
from lc_templates.core.hooks import clear_event_hooks, emit_event, register_event_hook
from lc_templates.core.logging import configure_logging_from_runtime, get_logger
from lc_templates.core.output import (
    build_agent_execution_result,
    render_agent_execution_result,
    render_classification_result,
    render_extraction_result,
    render_grounded_answer,
    render_health_report,
    render_knowledge_base_result,
    render_memory_operation_result,
    render_route_decision,
    render_task_result,
    render_workflow_execution_result,
)
from lc_templates.core.schemas import (
    AgentExecutionResult,
    ClassificationResult,
    ConfigWarning,
    ExecutionMetadata,
    ExtractionResult,
    GroundedAnswer,
    HealthReport,
    HealthSummary,
    KnowledgeBaseBuildResult,
    MemoryThreadOperationResult,
    ProviderStatus,
    ResultEnvelope,
    RouteDecision,
    TaskBundleResult,
    WorkflowExecutionResult,
)
from lc_templates.rag.indexing import build_vector_store, load_vector_store, resolve_collection_name
from lc_templates.rag.loaders import load_documents
from lc_templates.rag.pipeline import answer_with_rag, answer_with_structured_rag, render_rag_answer
from lc_templates.rag.retrievers import create_vector_retriever
from lc_templates.rag.splitters import split_documents
from lc_templates.workflows.router import route_decision, route_task

logger = get_logger(__name__)


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
        logger.info(
            "TemplateApp initialized. config_path=%s active_provider=%s",
            str(get_resolved_config_path(config_path)),
            self.settings.get_active_provider_name(),
        )
        emit_event(
            "app.initialized",
            message="TemplateApp initialized.",
            meta=ExecutionMetadata(
                provider_name=self.settings.get_active_provider_name(),
                operation="app_init",
            ),
            payload={"config_path": str(get_resolved_config_path(config_path))},
        )

    def on_event(self, callback):
        return register_event_hook(callback)

    def clear_event_hooks(self) -> None:
        clear_event_hooks()

    def _emit_result_event(self, name: str, result: ResultEnvelope, *, message: str) -> None:
        emit_event(
            name,
            message=message,
            trace_id=result.trace_id,
            meta=result.meta,
            payload={
                "status": result.status,
                "fallback_used": result.fallback_used,
                "error_reason": result.error_reason,
                "latency_ms": result.latency_ms,
            },
        )

    def _recommend_middleware_profile(self) -> tuple[str, str]:
        provider = self.settings.get_provider_definition(self.settings.get_active_provider_name())
        if provider.type == "ollama":
            if provider.reasoning_model and provider.reasoning_model != provider.chat_model:
                return (
                    "aggressive",
                    "The active provider is local and has a distinct reasoning model, "
                    "so aggressive middleware is affordable and can improve complex runs.",
                )
            return (
                "balanced",
                "The active provider is local, so balanced middleware is a good default "
                "without adding unnecessary switching behavior.",
            )
        return (
            "safe",
            "The active provider is remote, so safe middleware is recommended to limit "
            "cost, latency spikes, and risky tool behavior.",
        )

    def _finalize_result(
        self,
        result: ResultEnvelope,
        *,
        started_at: float,
        operation: str,
    ) -> ResultEnvelope:
        latency_ms = (perf_counter() - started_at) * 1000
        meta = result.meta.model_copy(update={"operation": result.meta.operation or operation})
        finalized = result.model_copy(
            update={
                "trace_id": uuid4().hex,
                "latency_ms": round(latency_ms, 3),
                "meta": meta,
            }
        )
        self._emit_result_event(
            f"{operation}.completed",
            finalized,
            message=f"{operation} completed.",
        )
        return finalized

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
        started_at = perf_counter()
        result = classify_text(text, labels)
        return self._finalize_result(result, started_at=started_at, operation="classify")

    def classify_label(self, text: str, labels: list[str]) -> str:
        return self.classify(text, labels).label

    def classify_text_result(self, text: str, labels: list[str], *, verbose: bool = False) -> str:
        return render_classification_result(
            self.classify(text, labels), detail_level="verbose" if verbose else "concise"
        )

    def extract(self, text: str) -> ExtractionResult:
        started_at = perf_counter()
        result = extract_entities(text)
        return self._finalize_result(result, started_at=started_at, operation="extract")

    def extract_text(self, text: str) -> str:
        return self.extract(text).summary

    def extract_display(self, text: str, *, verbose: bool = False) -> str:
        return render_extraction_result(
            self.extract(text), detail_level="verbose" if verbose else "concise"
        )

    def run_text_tasks(self, text: str) -> TaskBundleResult:
        started_at = perf_counter()
        result = run_text_tasks(text)
        return self._finalize_result(result, started_at=started_at, operation="tasks")

    def run_text_tasks_display(self, text: str, *, verbose: bool = False) -> str:
        return render_task_result(
            self.run_text_tasks(text), detail_level="verbose" if verbose else "concise"
        )

    def run_text_tasks_json(self, text: str) -> dict:
        return self.run_text_tasks(text).model_dump()

    def run(
        self,
        text: str,
        *,
        use_rag: bool = False,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        provider_name: str | None = None,
        k: int | None = None,
    ) -> WorkflowExecutionResult:
        started_at = perf_counter()
        logger.info("Running unified workflow. use_rag=%s input_length=%s", use_rag, len(text))
        emit_event(
            "run.started",
            message="Unified workflow started.",
            meta=ExecutionMetadata(
                provider_name=self.settings.get_active_provider_name(),
                operation="run",
            ),
            payload={"use_rag": use_rag, "input_length": len(text)},
        )
        route_result = self.route_decision(text)
        if route_result.name == "summarize":
            summary = self.summarize(text)
            result = WorkflowExecutionResult(
                route=route_result,
                text=summary,
                payload={"summary": summary},
                status=route_result.status,
                fallback_used=route_result.fallback_used,
                error_reason=route_result.error_reason,
                meta=route_result.meta.model_copy(update={"operation": "run"}),
            )
            return self._finalize_result(result, started_at=started_at, operation="run")
        if route_result.name == "extract":
            extraction = self.extract(text)
            result = WorkflowExecutionResult(
                route=route_result,
                text=render_extraction_result(extraction, detail_level="concise"),
                payload=extraction.model_dump(),
                status=extraction.status,
                fallback_used=route_result.fallback_used or extraction.fallback_used,
                error_reason=extraction.error_reason or route_result.error_reason,
                meta=extraction.meta.model_copy(update={"operation": "run"}),
            )
            return self._finalize_result(result, started_at=started_at, operation="run")
        if route_result.name == "rag":
            if use_rag:
                rag_result = self.ask_rag_structured(
                    text,
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                    provider_name=provider_name,
                    k=k,
                )
                result = WorkflowExecutionResult(
                    route=route_result,
                    text=render_grounded_answer(rag_result),
                    payload=rag_result.model_dump(),
                    status=rag_result.status,
                    fallback_used=route_result.fallback_used or rag_result.fallback_used,
                    error_reason=rag_result.error_reason or route_result.error_reason,
                    meta=rag_result.meta.model_copy(update={"operation": "run"}),
                )
            else:
                result = WorkflowExecutionResult(
                    route=route_result,
                    text=(
                        "RAG route selected. Build or load a knowledge base first, "
                        "then call ask_rag_rendered() or ask_rag_structured(), "
                        "or pass use_rag=True to run()."
                    ),
                    payload={"hint": "use_rag_api"},
                    status="partial",
                    fallback_used=True,
                    error_reason="rag_requires_knowledge_base",
                    meta=route_result.meta.model_copy(update={"operation": "run"}),
                )
            return self._finalize_result(result, started_at=started_at, operation="run")
        response = self.chat(text)
        result = WorkflowExecutionResult(
            route=route_result,
            text=response,
            payload={"text": response},
            status=route_result.status,
            fallback_used=route_result.fallback_used,
            error_reason=route_result.error_reason,
            meta=route_result.meta.model_copy(update={"operation": "run"}),
        )
        return self._finalize_result(result, started_at=started_at, operation="run")

    def run_display(
        self,
        text: str,
        *,
        verbose: bool = False,
        use_rag: bool = False,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        provider_name: str | None = None,
        k: int | None = None,
    ) -> str:
        return render_workflow_execution_result(
            self.run(
                text,
                use_rag=use_rag,
                persist_directory=persist_directory,
                collection_name=collection_name,
                provider_name=provider_name,
                k=k,
            ),
            detail_level="verbose" if verbose else "concise",
        )

    def route(self, text: str) -> str:
        return route_task(text)

    def route_decision(self, text: str) -> RouteDecision:
        started_at = perf_counter()
        result = route_decision(text)
        return self._finalize_result(result, started_at=started_at, operation="route")

    def route_name(self, text: str) -> str:
        return self.route_decision(text).name

    def route_result(self, text: str) -> RouteDecision:
        return self.route_decision(text)

    def route_display(self, text: str, *, verbose: bool = False) -> str:
        return render_route_decision(
            self.route_decision(text), detail_level="verbose" if verbose else "concise"
        )

    def agent(self, text: str) -> AgentExecutionResult:
        started_at = perf_counter()
        result = build_agent_execution_result(run_basic_agent(text))
        return self._finalize_result(result, started_at=started_at, operation="agent")

    def agent_text(self, text: str) -> str:
        return self.agent(text).final_text

    def agent_display(self, text: str, *, verbose: bool = False) -> str:
        return render_agent_execution_result(
            self.agent(text), detail_level="verbose" if verbose else "concise"
        )

    def memory_agent(self, thread_id: str, text: str) -> AgentExecutionResult:
        started_at = perf_counter()
        result = build_agent_execution_result(run_memory_agent(thread_id, text))
        return self._finalize_result(result, started_at=started_at, operation="memory_agent")

    def memory_agent_text(self, thread_id: str, text: str) -> str:
        return self.memory_agent(thread_id, text).final_text

    def memory_agent_display(self, thread_id: str, text: str, *, verbose: bool = False) -> str:
        return render_agent_execution_result(
            self.memory_agent(thread_id, text),
            detail_level="verbose" if verbose else "concise",
        )

    def _memory_storage_path(self) -> str:
        memory = self.settings.runtime.memory
        if memory.backend != "sqlite":
            return ""
        resolved = resolve_runtime_path(memory.sqlite_path)
        return str(resolved) if resolved else memory.sqlite_path

    def clear_memory_thread(self, thread_id: str) -> MemoryThreadOperationResult:
        started_at = perf_counter()
        checkpointer = build_memory_checkpointer()
        if not hasattr(checkpointer, "delete_thread"):
            result = MemoryThreadOperationResult(
                action="clear_memory_thread",
                thread_id=thread_id,
                backend=self.settings.runtime.memory.backend,
                storage_path=self._memory_storage_path(),
                status="unavailable",
                fallback_used=True,
                error_reason="memory_backend_missing_delete_thread",
                meta=ExecutionMetadata(operation="memory_admin"),
            )
            return self._finalize_result(result, started_at=started_at, operation="memory_admin")
        checkpointer.delete_thread(thread_id)
        result = MemoryThreadOperationResult(
            action="clear_memory_thread",
            thread_id=thread_id,
            backend=self.settings.runtime.memory.backend,
            storage_path=self._memory_storage_path(),
            meta=ExecutionMetadata(operation="memory_admin"),
        )
        return self._finalize_result(result, started_at=started_at, operation="memory_admin")

    def copy_memory_thread(
        self, source_thread_id: str, target_thread_id: str
    ) -> MemoryThreadOperationResult:
        started_at = perf_counter()
        checkpointer = build_memory_checkpointer()
        if not hasattr(checkpointer, "copy_thread"):
            result = MemoryThreadOperationResult(
                action="copy_memory_thread",
                thread_id=source_thread_id,
                target_thread_id=target_thread_id,
                backend=self.settings.runtime.memory.backend,
                storage_path=self._memory_storage_path(),
                status="unavailable",
                fallback_used=True,
                error_reason="memory_backend_missing_copy_thread",
                meta=ExecutionMetadata(operation="memory_admin"),
            )
            return self._finalize_result(result, started_at=started_at, operation="memory_admin")
        checkpointer.copy_thread(source_thread_id, target_thread_id)
        result = MemoryThreadOperationResult(
            action="copy_memory_thread",
            thread_id=source_thread_id,
            target_thread_id=target_thread_id,
            backend=self.settings.runtime.memory.backend,
            storage_path=self._memory_storage_path(),
            meta=ExecutionMetadata(operation="memory_admin"),
        )
        return self._finalize_result(result, started_at=started_at, operation="memory_admin")

    def prune_memory_threads(
        self, thread_ids: list[str], *, strategy: str = "keep_latest"
    ) -> MemoryThreadOperationResult:
        started_at = perf_counter()
        checkpointer = build_memory_checkpointer()
        if not hasattr(checkpointer, "prune"):
            result = MemoryThreadOperationResult(
                action="prune_memory_threads",
                thread_id=",".join(thread_ids),
                backend=self.settings.runtime.memory.backend,
                storage_path=self._memory_storage_path(),
                status="unavailable",
                fallback_used=True,
                error_reason="memory_backend_missing_prune",
                meta=ExecutionMetadata(operation="memory_admin"),
            )
            return self._finalize_result(result, started_at=started_at, operation="memory_admin")
        checkpointer.prune(thread_ids, strategy=strategy)
        result = MemoryThreadOperationResult(
            action="prune_memory_threads",
            thread_id=",".join(thread_ids),
            backend=self.settings.runtime.memory.backend,
            storage_path=self._memory_storage_path(),
            meta=ExecutionMetadata(
                operation="memory_admin",
                model_name=strategy,
            ),
        )
        return self._finalize_result(result, started_at=started_at, operation="memory_admin")

    def clear_memory_thread_display(self, thread_id: str, *, verbose: bool = False) -> str:
        return render_memory_operation_result(
            self.clear_memory_thread(thread_id),
            detail_level="verbose" if verbose else "concise",
        )

    def copy_memory_thread_display(
        self,
        source_thread_id: str,
        target_thread_id: str,
        *,
        verbose: bool = False,
    ) -> str:
        return render_memory_operation_result(
            self.copy_memory_thread(source_thread_id, target_thread_id),
            detail_level="verbose" if verbose else "concise",
        )

    def prune_memory_threads_display(
        self,
        thread_ids: list[str],
        *,
        strategy: str = "keep_latest",
        verbose: bool = False,
    ) -> str:
        return render_memory_operation_result(
            self.prune_memory_threads(thread_ids, strategy=strategy),
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
        started_at = perf_counter()
        logger.info("Indexing file. path=%s", path)
        documents = self.load_documents(path)
        chunks = self.split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        resolved_provider_name = provider_name or self.settings.get_active_provider_name()
        provider = self.settings.get_provider_definition(resolved_provider_name)
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
        result = KnowledgeBaseBuildResult(
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
            meta=ExecutionMetadata(
                provider_name=resolved_provider_name,
                model_name=provider.embedding_model,
                operation="index",
            ),
        )
        return self._finalize_result(result, started_at=started_at, operation="index")

    def index_display(
        self,
        path: str,
        *,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        provider_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        verbose: bool = False,
    ) -> str:
        return render_knowledge_base_result(
            self.index_file(
                path,
                persist_directory=persist_directory,
                collection_name=collection_name,
                provider_name=provider_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ),
            detail_level="verbose" if verbose else "concise",
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
        started_at = perf_counter()
        retriever = self.get_retriever(
            persist_directory=persist_directory,
            collection_name=collection_name,
            provider_name=provider_name,
            k=k,
        )
        result = answer_with_structured_rag(retriever, question)
        return self._finalize_result(result, started_at=started_at, operation="rag")

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
        logger.info("Running doctor health check.")
        emit_event(
            "doctor.started",
            message="Doctor health check started.",
            meta=ExecutionMetadata(
                provider_name=self.settings.get_active_provider_name(),
                operation="doctor",
            ),
        )
        resolved_path = get_resolved_config_path(self.config_path)
        providers: list[ProviderStatus] = []
        warnings: list[ConfigWarning] = []
        recommendations: list[str] = []
        recommended_profile, recommended_profile_reason = self._recommend_middleware_profile()
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
        middleware = self.settings.runtime.middleware
        memory = self.settings.runtime.memory
        if middleware.profile == "aggressive":
            recommendations.append(
                "Middleware profile is 'aggressive'; prefer this only when you have a clear "
                "reason to trade cost and latency for stronger reasoning behavior."
            )
        elif middleware.profile == "safe":
            recommendations.append(
                "Middleware profile is 'safe'; this is a good default for production-facing "
                "agents that should minimize risky tool behavior and protect common PII."
            )
        if memory.enabled and memory.backend == "memory":
            warnings.append(
                ConfigWarning(
                    code="memory_not_persistent",
                    message=(
                        "runtime.memory.backend is set to 'memory', so conversation history "
                        "will be lost when the Python process exits."
                    ),
                )
            )
            recommendations.append(
                "Use runtime.memory.backend=sqlite if you want memory-agent state to survive "
                "process restarts."
            )
        elif memory.enabled and memory.backend == "sqlite":
            resolved_memory_path = resolve_runtime_path(memory.sqlite_path)
            if resolved_memory_path is None:
                warnings.append(
                    ConfigWarning(
                        code="memory_sqlite_path_invalid",
                        message="runtime.memory.sqlite_path could not be resolved.",
                        severity="error",
                    )
                )
            else:
                recommendations.append(
                    "Memory persistence is enabled with SQLite. Keep the database path on "
                    "local durable storage if you rely on long-lived thread history."
                )
        if middleware.enabled and middleware.summarization.enabled:
            if middleware.summarization.keep_messages >= middleware.summarization.trigger_messages:
                warnings.append(
                    ConfigWarning(
                        code="middleware_summary_window_invalid",
                        message=(
                            "middleware.summarization.keep_messages should be smaller than "
                            "middleware.summarization.trigger_messages."
                        ),
                        severity="error",
                    )
                )
        if middleware.enabled and middleware.model_fallback_enabled:
            provider = self.settings.get_provider_definition(active_provider_name)
            if not provider.reasoning_model or provider.reasoning_model == provider.chat_model:
                warnings.append(
                    ConfigWarning(
                        code="middleware_model_fallback_inactive",
                        message=(
                            "model_fallback_enabled is on, but reasoning_model is empty or the "
                            "same as chat_model, so no effective fallback model is available."
                        ),
                    )
                )
                recommendations.append(
                    "Set providers.<active>.reasoning_model to a different model if you want "
                    "model fallback to have a real effect."
                )
        if middleware.enabled and middleware.dynamic_model_selection_enabled:
            provider = self.settings.get_provider_definition(active_provider_name)
            if not provider.reasoning_model or provider.reasoning_model == provider.chat_model:
                warnings.append(
                    ConfigWarning(
                        code="middleware_dynamic_model_selection_inactive",
                        message=(
                            "dynamic_model_selection_enabled is on, but reasoning_model is "
                            "empty or the same as chat_model, so the middleware cannot switch "
                            "to a separate model."
                        ),
                    )
                )
                recommendations.append(
                    "dynamic_model_selection_enabled is on, but it needs a distinct "
                    "reasoning_model to switch to."
                )
            else:
                recommendations.append(
                    "Dynamic model selection is enabled; tune "
                    "middleware.dynamic_model_selection_message_threshold if routing to the "
                    "reasoning model feels too eager or too conservative."
                )
        if middleware.profile != recommended_profile:
            recommendations.append(
                "Consider switching runtime.middleware.profile to "
                f"'{recommended_profile}' for this environment."
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
                if name == active_provider_name:
                    recommendations.append(
                        f"Set providers.{name}.api_key to a real key or environment variable "
                        "before making remote calls."
                    )
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
                if name == active_provider_name:
                    recommendations.append(
                        f"Set providers.{name}.embedding_model if you want to use indexing or RAG."
                    )
            elif provider.embedding_dimensions is None:
                provider_warnings.append(
                    "embedding_dimensions is not configured. "
                    "Default collection isolation will be less explicit."
                )
                recommendations.append(
                    f"Set providers.{name}.embedding_dimensions to make vector collections "
                    "more self-describing and avoid accidental mixing."
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

        ready = not any(item.severity == "error" for item in warnings)
        report = HealthReport(
            package_version=self.version(),
            active_provider=active_provider_name,
            rerank_provider=rerank_provider_name,
            config_path=str(Path(resolved_path)),
            default_collection_name=self.settings.runtime.default_collection_name,
            default_persist_directory=str(
                resolve_runtime_path(self.settings.runtime.default_persist_directory)
            ),
            ready=ready,
            warnings=warnings,
            recommendations=list(dict.fromkeys(recommendations)),
            recommended_middleware_profile=recommended_profile,
            recommended_middleware_reason=recommended_profile_reason,
            summary=HealthSummary(
                ready=ready,
                provider_count=len(providers),
                ready_provider_count=sum(1 for provider in providers if provider.ready),
                warning_count=len(warnings),
            ),
            providers=providers,
        )
        emit_event(
            "doctor.completed",
            message="Doctor health check completed.",
            meta=ExecutionMetadata(
                provider_name=active_provider_name,
                operation="doctor",
            ),
            payload={
                "ready": report.ready,
                "warning_count": report.summary.warning_count,
                "recommended_middleware_profile": report.recommended_middleware_profile,
            },
        )
        return report

    def doctor_summary(self) -> HealthSummary:
        return self.doctor().summary

    def doctor_recommendations(self) -> list[str]:
        return self.doctor().recommendations

    def doctor_recommended_profile(self) -> str:
        return self.doctor().recommended_middleware_profile

    def doctor_display(self, *, verbose: bool = False) -> str:
        return render_health_report(self.doctor(), detail_level="verbose" if verbose else "concise")

    def init_config(self, destination: str | None = None, *, overwrite: bool = False) -> str:
        return str(scaffold_config(destination, overwrite=overwrite))


def create_app(config_path: str | None = None) -> TemplateApp:
    return TemplateApp(config_path=config_path)
