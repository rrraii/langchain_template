from typing import Any, Literal

from pydantic import BaseModel, Field

ResultStatus = Literal["ok", "partial", "unavailable", "error"]


class ConfigWarning(BaseModel):
    code: str = Field(description="Stable warning code")
    message: str = Field(description="Human-readable warning message")
    severity: str = Field(default="warning", description="Severity level")


class ExecutionMetadata(BaseModel):
    provider_name: str = Field(default="", description="Resolved provider name")
    model_name: str = Field(default="", description="Resolved model name")
    operation: str = Field(default="", description="Logical operation name")


class HookEvent(BaseModel):
    name: str = Field(description="Stable event name")
    level: str = Field(default="INFO", description="Event severity level")
    message: str = Field(default="", description="Human-readable event summary")
    trace_id: str = Field(default="", description="Optional trace id for correlation")
    meta: ExecutionMetadata = Field(
        default_factory=ExecutionMetadata,
        description="Execution metadata attached to the event",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional machine-friendly event payload",
    )


class ResultEnvelope(BaseModel):
    trace_id: str = Field(default="", description="Per-result trace id for correlation")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Execution latency in milliseconds")
    status: ResultStatus = Field(default="ok", description="High-level execution status")
    fallback_used: bool = Field(
        default=False, description="Whether a fallback path was used to produce the result"
    )
    error_reason: str = Field(
        default="", description="Short machine-friendly error reason when the result degrades"
    )
    meta: ExecutionMetadata = Field(
        default_factory=ExecutionMetadata,
        description="Execution metadata for downstream tracing and inspection",
    )


class CitationItem(BaseModel):
    marker: str = Field(default="", description="Short citation marker such as [1]")
    source: str = Field(default="", description="Source identifier such as a file name")
    snippet: str = Field(default="", description="Short supporting snippet")


class ToolCallRecord(BaseModel):
    name: str = Field(default="", description="Tool name")
    call_id: str = Field(default="", description="Provider or framework tool call id")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool call arguments")


class CitationAnswer(ResultEnvelope):
    answer: str = Field(description="Final answer")
    citations: list[str] = Field(default_factory=list, description="Short supporting references")
    citation_items: list[CitationItem] = Field(
        default_factory=list,
        description="Normalized citation objects for downstream rendering or inspection",
    )


class GroundedAnswer(CitationAnswer):
    answer: str = Field(description="Final grounded answer")
    grounded: bool = Field(
        default=True, description="Whether the answer is supported by the retrieved context"
    )


class ClassificationResult(ResultEnvelope):
    label: str = Field(description="Chosen label")
    reason: str = Field(description="Short reason for the classification")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )


class ExtractionResult(ResultEnvelope):
    entities: list[str] = Field(
        default_factory=list, description="Entities extracted from the text"
    )
    summary: str = Field(default="", description="One-sentence summary")


class AgentExecutionResult(ResultEnvelope):
    final_text: str = Field(default="", description="Final assistant-facing text")
    used_tools: list[str] = Field(
        default_factory=list, description="Tool names used during the run"
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list, description="Normalized tool call records"
    )
    tool_call_count: int = Field(default=0, description="Total number of tool calls")
    raw: dict[str, Any] = Field(default_factory=dict, description="Raw agent result for debugging")


class KnowledgeBaseBuildResult(ResultEnvelope):
    source_path: str = Field(description="Input file path used to build the knowledge base")
    persist_directory: str = Field(description="Persist directory for the vector store")
    collection_name: str = Field(description="Collection name used for indexing")
    provider_name: str = Field(default="", description="Provider name used for embeddings")
    embedding_model: str = Field(default="", description="Embedding model used for indexing")
    embedding_dimensions: int | None = Field(
        default=None, description="Configured embedding vector dimensions"
    )
    document_count: int = Field(default=0, description="Number of loaded source documents")
    chunk_count: int = Field(default=0, description="Number of generated chunks")


class MemoryThreadOperationResult(ResultEnvelope):
    action: str = Field(default="", description="Memory operation name")
    thread_id: str = Field(default="", description="Primary thread id")
    target_thread_id: str = Field(default="", description="Optional target thread id")
    backend: str = Field(default="", description="Configured memory backend")
    storage_path: str = Field(default="", description="Resolved storage path when applicable")


class ProviderStatus(BaseModel):
    name: str = Field(description="Provider name")
    enabled: bool = Field(description="Whether the provider is enabled")
    provider_type: str = Field(description="Provider type")
    has_api_key: bool = Field(description="Whether the provider appears to have a usable API key")
    base_url: str = Field(default="", description="Configured provider base URL")
    chat_model: str = Field(default="", description="Configured chat model")
    embedding_model: str = Field(default="", description="Configured embedding model")
    embedding_dimensions: int | None = Field(
        default=None, description="Configured embedding vector dimensions"
    )
    ready: bool = Field(default=False, description="Whether the provider appears ready to serve")
    warnings: list[str] = Field(
        default_factory=list, description="Provider-specific readiness warnings"
    )


class HealthSummary(BaseModel):
    ready: bool = Field(default=False, description="Whether the setup is ready overall")
    provider_count: int = Field(default=0, description="Total configured providers")
    ready_provider_count: int = Field(default=0, description="Providers that appear ready")
    warning_count: int = Field(default=0, description="Total warning count")


class HealthReport(BaseModel):
    package_version: str = Field(description="Installed package version")
    active_provider: str = Field(description="Active provider name")
    rerank_provider: str = Field(description="Rerank provider name")
    config_path: str = Field(description="Resolved config path")
    default_collection_name: str = Field(description="Default collection name")
    default_persist_directory: str = Field(description="Default persist directory")
    ready: bool = Field(default=False, description="Whether the current setup appears usable")
    warnings: list[ConfigWarning] = Field(
        default_factory=list, description="High-level configuration warnings"
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Concrete next-step recommendations derived from the health check",
    )
    recommended_middleware_profile: str = Field(
        default="balanced",
        description="Suggested middleware profile for the current environment",
    )
    recommended_middleware_reason: str = Field(
        default="",
        description="Short reason explaining the suggested middleware profile",
    )
    summary: HealthSummary = Field(
        default_factory=HealthSummary,
        description="Short aggregate health summary",
    )
    providers: list[ProviderStatus] = Field(
        default_factory=list, description="Provider health summary"
    )


class RouteDecision(ResultEnvelope):
    route: str = Field(default="route:chat", description="Canonical route string")
    name: str = Field(default="chat", description="Normalized route name")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    reason: str = Field(default="", description="Short reason for the routing decision")


class TaskBundleResult(ResultEnvelope):
    summary: str = Field(default="", description="Summary of the input text")
    route: RouteDecision = Field(default_factory=RouteDecision, description="Route decision")
    classification: ClassificationResult = Field(
        default_factory=lambda: ClassificationResult(label="", reason="", confidence=0.0),
        description="Classification result",
    )
    extraction: ExtractionResult = Field(
        default_factory=ExtractionResult, description="Extraction result"
    )


class WorkflowExecutionResult(ResultEnvelope):
    route: RouteDecision = Field(default_factory=RouteDecision, description="Route decision")
    text: str = Field(default="", description="Human-readable result text")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured payload for downstream programmatic consumption",
    )
