from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

ProviderType = Literal["openai_compatible", "ollama"]
ResponseFormat = Literal["text", "markdown"]
AnswerStyle = Literal["concise", "balanced", "detailed"]
OutputMode = Literal["concise", "verbose", "json"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
MiddlewareProfile = Literal["safe", "balanced", "aggressive", "custom"]


class RuntimeSettings(BaseModel):
    active_provider: str = Field(default="qwen")
    rerank_provider: str | None = Field(default=None)
    http_proxy: str | None = Field(default=None)
    https_proxy: str | None = Field(default=None)
    default_collection_name: str = Field(default="demo_collection")
    default_persist_directory: str = Field(default="data/index/chroma")
    top_k: int = Field(default=4)
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)
    hybrid_rrf_k: int = Field(default=60, ge=1, le=200)
    response_language: str = Field(default="zh-CN")
    response_format: ResponseFormat = Field(default="markdown")
    answer_style: AnswerStyle = Field(default="balanced")
    default_output_mode: OutputMode = Field(default="concise")
    log_level: LogLevel = Field(default="INFO")
    third_party_log_level: LogLevel = Field(default="WARNING")
    log_file: str | None = Field(default=None)
    max_citations: int = Field(default=3, ge=1, le=10)
    routing_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    rag_no_answer_message: str = Field(
        default="I cannot answer confidently from the provided context."
    )
    middleware: MiddlewareSettings = Field(default_factory=lambda: MiddlewareSettings())


class PIIMiddlewareSettings(BaseModel):
    enabled: bool = Field(default=False)
    pii_types: list[str] = Field(default_factory=lambda: ["email", "url"])
    strategy: Literal["block", "redact", "mask", "hash"] = Field(default="redact")
    apply_to_input: bool = Field(default=True)
    apply_to_output: bool = Field(default=False)
    apply_to_tool_results: bool = Field(default=False)


class SummarizationMiddlewareSettings(BaseModel):
    enabled: bool = Field(default=True)
    trigger_messages: int = Field(default=24, ge=4, le=200)
    keep_messages: int = Field(default=12, ge=2, le=100)

    @model_validator(mode="after")
    def validate_window(self) -> SummarizationMiddlewareSettings:
        if self.keep_messages >= self.trigger_messages:
            raise ValueError("keep_messages must be smaller than trigger_messages")
        return self


class MiddlewareSettings(BaseModel):
    profile: MiddlewareProfile = Field(default="balanced")
    enabled: bool = Field(default=True)
    tool_call_limit_enabled: bool = Field(default=True)
    tool_call_limit: int = Field(default=6, ge=1, le=100)
    model_fallback_enabled: bool = Field(default=False)
    dynamic_model_selection_enabled: bool = Field(default=False)
    dynamic_model_selection_message_threshold: int = Field(default=800, ge=50, le=20000)
    pii: PIIMiddlewareSettings = Field(default_factory=PIIMiddlewareSettings)
    summarization: SummarizationMiddlewareSettings = Field(
        default_factory=SummarizationMiddlewareSettings
    )

    @model_validator(mode="before")
    @classmethod
    def apply_profile_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        profile = data.get("profile", "balanced")
        presets = {
            "safe": {
                "enabled": True,
                "tool_call_limit_enabled": True,
                "tool_call_limit": 4,
                "model_fallback_enabled": False,
                "dynamic_model_selection_enabled": False,
                "dynamic_model_selection_message_threshold": 1000,
                "pii": {
                    "enabled": True,
                    "pii_types": ["email", "url"],
                    "strategy": "redact",
                    "apply_to_input": True,
                    "apply_to_output": False,
                    "apply_to_tool_results": False,
                },
                "summarization": {
                    "enabled": True,
                    "trigger_messages": 20,
                    "keep_messages": 10,
                },
            },
            "balanced": {
                "enabled": True,
                "tool_call_limit_enabled": True,
                "tool_call_limit": 6,
                "model_fallback_enabled": False,
                "dynamic_model_selection_enabled": False,
                "dynamic_model_selection_message_threshold": 800,
                "pii": {
                    "enabled": False,
                    "pii_types": ["email", "url"],
                    "strategy": "redact",
                    "apply_to_input": True,
                    "apply_to_output": False,
                    "apply_to_tool_results": False,
                },
                "summarization": {
                    "enabled": True,
                    "trigger_messages": 24,
                    "keep_messages": 12,
                },
            },
            "aggressive": {
                "enabled": True,
                "tool_call_limit_enabled": True,
                "tool_call_limit": 10,
                "model_fallback_enabled": True,
                "dynamic_model_selection_enabled": True,
                "dynamic_model_selection_message_threshold": 500,
                "pii": {
                    "enabled": False,
                    "pii_types": ["email", "url"],
                    "strategy": "redact",
                    "apply_to_input": True,
                    "apply_to_output": False,
                    "apply_to_tool_results": False,
                },
                "summarization": {
                    "enabled": True,
                    "trigger_messages": 30,
                    "keep_messages": 12,
                },
            },
        }
        if profile == "custom":
            return data
        preset = presets.get(profile)
        if not preset:
            return data

        merged = {**preset, **data}
        merged["pii"] = {**preset["pii"], **data.get("pii", {})}
        merged["summarization"] = {
            **preset["summarization"],
            **data.get("summarization", {}),
        }
        return merged


class ProviderSettings(BaseModel):
    type: ProviderType = Field(default="openai_compatible")
    enabled: bool = Field(default=True)
    api_key: str = Field(default="")
    base_url: str = Field(default="")
    chat_model: str = Field(default="")
    reasoning_model: str = Field(default="")
    embedding_model: str = Field(default="")
    embedding_dimensions: int | None = Field(default=None, ge=1)
    rerank_model: str = Field(default="")
    temperature: float = Field(default=0.1)
    request_timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)

    def has_placeholder_api_key(self) -> bool:
        if not self.api_key:
            return True
        normalized = self.api_key.strip().lower()
        placeholders = {
            "your_openai_api_key",
            "your_dashscope_api_key",
            "your_deepseek_api_key",
            "dashscope_api_key",
        }
        return normalized in placeholders

    @model_validator(mode="after")
    def validate_provider(self) -> ProviderSettings:
        if self.type == "openai_compatible" and not self.base_url:
            raise ValueError("openai_compatible provider must define base_url")
        if self.type == "ollama" and not self.chat_model:
            raise ValueError("ollama provider must define chat_model")
        return self


class ProvidersSettings(BaseModel):
    openai: ProviderSettings = Field(
        default_factory=lambda: ProviderSettings(
            type="openai_compatible",
            base_url="https://api.openai.com/v1",
            chat_model="gpt-4.1-mini",
            reasoning_model="gpt-5-mini",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
        )
    )
    qwen: ProviderSettings = Field(
        default_factory=lambda: ProviderSettings(
            type="openai_compatible",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            chat_model="qwen3.5-plus",
            reasoning_model="deepseek-r1",
            embedding_model="text-embedding-v1",
            embedding_dimensions=1536,
            rerank_model="gte-rerank-v2",
        )
    )
    deepseek: ProviderSettings = Field(
        default_factory=lambda: ProviderSettings(
            type="openai_compatible",
            base_url="https://api.deepseek.com/v1",
            chat_model="deepseek-chat",
            reasoning_model="deepseek-reasoner",
        )
    )
    ollama: ProviderSettings = Field(
        default_factory=lambda: ProviderSettings(
            type="ollama",
            base_url="http://localhost:11434",
            chat_model="qwen2.5:7b",
            reasoning_model="qwen2.5:7b",
            embedding_model="nomic-embed-text",
            embedding_dimensions=768,
        )
    )


class AppSettings(BaseModel):
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)

    def get_provider_definition(self, provider_name: str) -> ProviderSettings:
        providers_dict = self.providers.model_dump()
        if provider_name not in providers_dict:
            raise KeyError(f"Provider not found: {provider_name}")
        return getattr(self.providers, provider_name)

    def get_provider(self, provider_name: str) -> ProviderSettings:
        provider = self.get_provider_definition(provider_name)
        if not provider.enabled:
            raise ValueError(f"Provider is disabled: {provider_name}")
        if provider.type == "openai_compatible" and provider.has_placeholder_api_key():
            raise ValueError(f"Provider '{provider_name}' must define a valid api_key")
        return provider

    def get_active_provider_name(self) -> str:
        return self.runtime.active_provider

    def get_active_provider(self) -> ProviderSettings:
        return self.get_provider(self.runtime.active_provider)

    def get_rerank_provider_name(self) -> str:
        return self.runtime.rerank_provider or self.runtime.active_provider

    def get_rerank_provider(self) -> ProviderSettings:
        return self.get_provider(self.get_rerank_provider_name())


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_config_path() -> Path:
    env_path = os.getenv("LC_TEMPLATES_CONFIG")
    if env_path:
        return Path(env_path)
    return _project_root() / "config" / "config.yaml"


def _example_config_path() -> Path:
    return _project_root() / "config" / "config.example.yaml"


def get_resolved_config_path(config_path: str | None = None) -> Path:
    return Path(config_path) if config_path else _default_config_path()


def resolve_runtime_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _project_root() / path


def scaffold_config(destination: str | None = None, *, overwrite: bool = False) -> Path:
    target_path = Path(destination) if destination else _default_config_path()
    source_path = _example_config_path()
    if not source_path.exists():
        raise FileNotFoundError(f"Example config file not found: {source_path}")
    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Config file already exists: {target_path}. Use overwrite=True to replace it."
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    return target_path


def _resolve_env_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return value

    if stripped.startswith("${") and stripped.endswith("}"):
        env_name = stripped[2:-1].strip()
        return os.getenv(env_name, value)

    if stripped in os.environ:
        return os.environ[stripped]

    return value


def _resolve_env_placeholders(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _resolve_env_placeholders(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_resolve_env_placeholders(item) for item in data]
    return _resolve_env_value(data)


def _load_yaml_config(config_path: str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else _default_config_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            "Please create config/config.yaml from config/config.example.yaml first."
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format: {path}. Top-level value must be a mapping.")
    return _resolve_env_placeholders(data)


def _apply_runtime_environment(settings: AppSettings) -> None:
    runtime = settings.runtime
    proxy_pairs = {
        "HTTP_PROXY": runtime.http_proxy,
        "HTTPS_PROXY": runtime.https_proxy,
    }
    for env_name, value in proxy_pairs.items():
        if value:
            os.environ[env_name] = value
        else:
            os.environ.pop(env_name, None)


@lru_cache(maxsize=4)
def get_settings(config_path: str | None = None) -> AppSettings:
    data = _load_yaml_config(config_path)
    settings = AppSettings(**data)
    _apply_runtime_environment(settings)
    return settings
