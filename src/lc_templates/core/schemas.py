from typing import Any

from pydantic import BaseModel, Field


class ConfigWarning(BaseModel):
    code: str = Field(description="Stable warning code")
    message: str = Field(description="Human-readable warning message")
    severity: str = Field(default="warning", description="Severity level")


class CitationAnswer(BaseModel):
    answer: str = Field(description="Final answer")
    citations: list[str] = Field(default_factory=list, description="Short supporting references")


class GroundedAnswer(BaseModel):
    answer: str = Field(description="Final grounded answer")
    citations: list[str] = Field(default_factory=list, description="Short supporting references")
    grounded: bool = Field(
        default=True, description="Whether the answer is supported by the retrieved context"
    )


class ClassificationResult(BaseModel):
    label: str = Field(description="Chosen label")
    reason: str = Field(description="Short reason for the classification")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )


class ExtractionResult(BaseModel):
    entities: list[str] = Field(
        default_factory=list, description="Entities extracted from the text"
    )
    summary: str = Field(default="", description="One-sentence summary")
    status: str = Field(default="ok", description="Extraction status such as ok or unavailable")
    fallback_used: bool = Field(
        default=False, description="Whether a fallback path was used to produce the result"
    )
    error_reason: str = Field(
        default="", description="Short machine-friendly error reason when extraction degrades"
    )


class AgentExecutionResult(BaseModel):
    final_text: str = Field(default="", description="Final assistant-facing text")
    used_tools: list[str] = Field(
        default_factory=list, description="Tool names used during the run"
    )
    tool_call_count: int = Field(default=0, description="Total number of tool calls")
    raw: dict[str, Any] = Field(default_factory=dict, description="Raw agent result for debugging")


class KnowledgeBaseBuildResult(BaseModel):
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
    providers: list[ProviderStatus] = Field(
        default_factory=list, description="Provider health summary"
    )
