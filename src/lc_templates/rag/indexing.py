from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from lc_templates.core.config import AppSettings, get_settings, resolve_runtime_path
from lc_templates.core.logging import get_logger
from lc_templates.core.models import build_embeddings

logger = get_logger(__name__)


def _slugify_embedding_value(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or "default"


def resolve_collection_name(
    collection_name: str | None = None,
    *,
    provider_name: str | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
    settings: AppSettings | None = None,
) -> str:
    if collection_name:
        return collection_name

    resolved_settings = settings or get_settings()
    resolved_provider_name = provider_name or resolved_settings.get_active_provider_name()
    provider = resolved_settings.get_provider_definition(resolved_provider_name)
    model_name = embedding_model or provider.embedding_model or "embedding"
    dimension = embedding_dimensions or provider.embedding_dimensions
    suffix_parts = [resolved_provider_name, _slugify_embedding_value(model_name)]
    if dimension:
        suffix_parts.append(f"{dimension}d")
    suffix = "__".join(suffix_parts)
    return f"{resolved_settings.runtime.default_collection_name}__{suffix}"


def build_vector_store(
    documents: list[Document],
    persist_directory: str | None = None,
    collection_name: str | None = None,
    provider_name: str | None = None,
) -> Chroma:
    settings = get_settings()
    target_directory = resolve_runtime_path(
        persist_directory or settings.runtime.default_persist_directory
    )
    if target_directory is None:
        raise ValueError("persist_directory could not be resolved")
    Path(target_directory).mkdir(parents=True, exist_ok=True)
    embeddings = build_embeddings(provider_name=provider_name)
    resolved_collection_name = resolve_collection_name(
        collection_name,
        provider_name=provider_name,
        settings=settings,
    )
    logger.info(
        "Building vector store. persist_directory=%s collection_name=%s provider=%s",
        str(target_directory),
        resolved_collection_name,
        provider_name or settings.get_active_provider_name(),
    )
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(target_directory),
        collection_name=resolved_collection_name,
    )


def load_vector_store(
    persist_directory: str | None = None,
    collection_name: str | None = None,
    provider_name: str | None = None,
) -> Chroma:
    settings = get_settings()
    embeddings = build_embeddings(provider_name=provider_name)
    target_directory = resolve_runtime_path(
        persist_directory or settings.runtime.default_persist_directory
    )
    if target_directory is None:
        raise ValueError("persist_directory could not be resolved")
    resolved_collection_name = resolve_collection_name(
        collection_name,
        provider_name=provider_name,
        settings=settings,
    )
    logger.info(
        "Loading vector store. persist_directory=%s collection_name=%s provider=%s",
        str(target_directory),
        resolved_collection_name,
        provider_name or settings.get_active_provider_name(),
    )
    return Chroma(
        persist_directory=str(target_directory),
        embedding_function=embeddings,
        collection_name=resolved_collection_name,
    )
