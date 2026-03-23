from __future__ import annotations

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

from lc_templates.core.config import ProviderSettings, get_settings


def _resolve_provider(provider_name: str | None = None) -> tuple[str, ProviderSettings]:
    """Resolve the configured provider for real model calls."""
    settings = get_settings()
    name = provider_name or settings.get_active_provider_name()
    return name, settings.get_provider(name)


def get_active_provider_name() -> str:
    settings = get_settings()
    return settings.get_active_provider_name()


def build_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    provider_name: str | None = None,
):
    _, provider = _resolve_provider(provider_name)
    model_name = model or provider.chat_model
    final_temperature = provider.temperature if temperature is None else temperature

    if provider.type == "ollama":
        return ChatOllama(
            model=model_name,
            base_url=provider.base_url,
            temperature=final_temperature,
        )

    return ChatOpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
        model=model_name,
        temperature=final_temperature,
        timeout=provider.request_timeout,
        max_retries=provider.max_retries,
    )


def build_reasoning_model(provider_name: str | None = None):
    _, provider = _resolve_provider(provider_name)
    model_name = provider.reasoning_model or provider.chat_model

    if provider.type == "ollama":
        return ChatOllama(
            model=model_name,
            base_url=provider.base_url,
            temperature=provider.temperature,
        )

    return ChatOpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
        model=model_name,
        temperature=provider.temperature,
        timeout=provider.request_timeout,
        max_retries=provider.max_retries,
    )


def build_embeddings(model: str | None = None, provider_name: str | None = None):
    provider_key, provider = _resolve_provider(provider_name)
    model_name = model or provider.embedding_model
    if not model_name:
        raise ValueError("Provider must define embedding_model for embedding workflows.")

    if provider.type == "ollama":
        return OllamaEmbeddings(model=model_name, base_url=provider.base_url)

    # Some OpenAI-compatible providers reject tokenized embedding payloads and only
    # accept raw strings or list[str]. Disable tiktoken-based preprocessing for them.
    embedding_kwargs = {}
    if provider_key != "openai":
        embedding_kwargs = {
            "tiktoken_enabled": False,
            "check_embedding_ctx_length": False,
        }

    return OpenAIEmbeddings(
        model=model_name,
        api_key=provider.api_key,
        base_url=provider.base_url,
        request_timeout=provider.request_timeout,
        max_retries=provider.max_retries,
        **embedding_kwargs,
    )


def build_openai_compatible_client(
    provider_name: str | None = None, for_rerank: bool = False
) -> OpenAI:
    """Build a low-level OpenAI-compatible client for rerank and provider-specific APIs."""
    settings = get_settings()
    target_name = provider_name or (
        settings.get_rerank_provider_name() if for_rerank else settings.get_active_provider_name()
    )
    provider = settings.get_provider(target_name)

    if provider.type != "openai_compatible":
        raise ValueError(
            f"Provider '{target_name}' must be openai_compatible to build an OpenAI client."
        )

    return OpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
        timeout=provider.request_timeout,
        max_retries=provider.max_retries,
    )
