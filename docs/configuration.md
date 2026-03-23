# Configuration

Runtime behavior is controlled by `config/config.yaml`.

## How to read these settings

Not every config field has the same enforcement strength. In this project, settings generally fall into three categories:

- Hard enforcement: directly changes code paths, validation, routing, retrieval, client options, or final rendering.
- Soft preference: mainly injected into prompts as guidance for the model. These improve consistency, but weaker models may not follow them perfectly.
- Default override: provides a default value that can still be overridden per call, by CLI flags, or by higher-level APIs.

When a field is not fully strict, this documentation calls that out explicitly.

## File layout

```yaml
runtime:
  ...

providers:
  openai:
    ...
  qwen:
    ...
  deepseek:
    ...
  ollama:
    ...
```

The top-level object must be a mapping with:

- `runtime`: global application behavior
- `providers`: model provider definitions

## Environment variable support

String values can be resolved from environment variables in two forms:

```yaml
api_key: DASHSCOPE_API_KEY
```

```yaml
api_key: ${DASHSCOPE_API_KEY}
```

Both forms will try to read `DASHSCOPE_API_KEY` from the current process environment.

## Runtime section

### `active_provider`

- Type: `str`
- Default: `"qwen"`
- Enforcement: hard enforcement
- Meaning: provider name used for normal chat, agent, and structured-output calls
- Common values: `openai`, `qwen`, `deepseek`, `ollama`

### `rerank_provider`

- Type: `str | null`
- Default: `null`
- Enforcement: hard enforcement
- Meaning: provider name used for rerank-related calls
- Behavior: if omitted, the framework falls back to `active_provider`

### `http_proxy`

- Type: `str | null`
- Default: `null`
- Enforcement: hard enforcement
- Meaning: HTTP proxy URL written to the `HTTP_PROXY` environment variable during config loading
- Example: `http://127.0.0.1:7890`
- Note: set to `null` to disable it

### `https_proxy`

- Type: `str | null`
- Default: `null`
- Enforcement: hard enforcement
- Meaning: HTTPS proxy URL written to the `HTTPS_PROXY` environment variable during config loading
- Example: `http://127.0.0.1:7890`
- Note: set to `null` to disable it

### `default_collection_name`

- Type: `str`
- Default: `"demo_collection"`
- Enforcement: default override
- Meaning: base Chroma collection name used by indexing and retrieval helpers
- Behavior: when you do not pass an explicit `collection_name`, the framework derives the effective collection as `default_collection_name__provider__embedding-model__dimension`
- Example: `demo_collection__qwen__text-embedding-v1__1536d`
- Note: can still be overridden per indexing or retrieval call

### `default_persist_directory`

- Type: `str`
- Default: `"data/index/chroma"`
- Enforcement: default override
- Meaning: default on-disk directory for vector store persistence
- Behavior: relative paths are resolved from the project root, not from the current working directory
- Note: can be overridden per indexing or retrieval call

### `top_k`

- Type: `int`
- Default: `4`
- Enforcement: default override
- Meaning: default number of retrieved documents for RAG
- Typical range: `3` to `8`
- Note: can be overridden when creating or using retrievers

### `chunk_size`

- Type: `int`
- Default: `500`
- Allowed range: `100` to `4000`
- Enforcement: default override
- Meaning: target chunk length for document splitting
- Note: can be overridden per split or indexing call

### `chunk_overlap`

- Type: `int`
- Default: `100`
- Allowed range: `0` to `1000`
- Enforcement: default override with validation warning
- Meaning: overlap between adjacent chunks
- Recommendation: keep this smaller than `chunk_size`
- Note: the doctor report warns if `chunk_overlap >= chunk_size`

### `hybrid_rrf_k`

- Type: `int`
- Default: `60`
- Allowed range: `1` to `200`
- Enforcement: hard enforcement where hybrid retrieval is used
- Meaning: reciprocal-rank-fusion smoothing parameter for hybrid retrieval

### `response_language`

- Type: `str`
- Default: `"zh-CN"`
- Enforcement: soft preference
- Meaning: preferred response language injected into prompt instructions
- Examples: `zh-CN`, `en`, `ja`
- Note: this guides model output, but does not hard-translate post-processing results

### `response_format`

- Type: `"text" | "markdown"`
- Default: `"markdown"`
- Enforcement: mostly soft preference, partially hard in RAG rendering
- Meaning: preferred answer formatting style in prompts
- Note: this strongly affects prompt instructions, but currently only has strict rendering impact in some citation-based RAG outputs

### `answer_style`

- Type: `"concise" | "balanced" | "detailed"`
- Default: `"balanced"`
- Enforcement: soft preference
- Meaning: default verbosity preference injected into prompts
- Note: this is not a hard truncation or expansion rule; the model may still vary

### `default_output_mode`

- Type: `"concise" | "verbose" | "json"`
- Default: `"concise"`
- Enforcement: default override
- Meaning: default display mode for examples and supported CLI commands
- Override: can be overridden per run with `--output`
- Note: this affects examples and CLI display, not core chain logic

### `log_level`

- Type: `"DEBUG" | "INFO" | "WARNING" | "ERROR"`
- Default: `"INFO"`
- Enforcement: hard enforcement
- Meaning: global application log level configured when `TemplateApp` starts
- Recommendation:
  - use `DEBUG` when diagnosing fallback paths, schema drift, timeouts, or collection resolution
  - use `INFO` for normal development

### `third_party_log_level`

- Type: `"DEBUG" | "INFO" | "WARNING" | "ERROR"`
- Default: `"WARNING"`
- Enforcement: hard enforcement
- Meaning: log level used for noisy third-party SDK loggers such as `openai`, `httpx`, and `httpcore`
- Recommendation: keep this at `WARNING` or `ERROR` during normal use so retry noise does not dominate the console

### `log_file`

- Type: `str | null`
- Default: `null` in code, repository config sets `logs/lc_templates.log`
- Enforcement: hard enforcement
- Meaning: optional file path for persisted application logs
- Behavior: when set, the framework writes logs both to the console and to this file
- Path rule: relative paths are resolved from the project root, not from the current working directory

### `max_citations`

- Type: `int`
- Default: `3`
- Allowed range: `1` to `10`
- Enforcement: mixed
- Meaning: maximum number of citations rendered in grounded RAG answers
- Note: this is a hard cap in rendering and filtering, and also a soft prompt hint elsewhere

### `routing_confidence_threshold`

- Type: `float`
- Default: `0.55`
- Allowed range: `0.0` to `1.0`
- Enforcement: hard enforcement
- Meaning: fallback threshold used by router classification
- Behavior: if confidence is below this threshold, the router falls back to `chat`

### `rag_no_answer_message`

- Type: `str`
- Default: `"I cannot answer confidently from the provided context."`
- Enforcement: mixed
- Meaning: fallback text returned when retrieved context is insufficient
- Note: this is used as a hard fallback in RAG post-processing and also injected into prompts

### `middleware`

- Type: object
- Default: enabled with conservative defaults
- Enforcement: hard enforcement for agent construction
- Meaning: controls LangChain 1.2 agent middleware attached to `create_agent(...)`
- Current middleware support:
  - tool call limiting
  - optional dynamic model selection
  - optional model fallback
  - optional PII protection
  - optional memory summarization for memory-enabled agents

#### `middleware.profile`

- Type: `"safe" | "balanced" | "aggressive" | "custom"`
- Default: `"balanced"`
- Meaning: preset that seeds middleware defaults before individual overrides are applied
- Presets:
  - `safe`: lower tool-call budget, PII protection on, conservative switching
  - `balanced`: practical default for most local and remote agent runs
  - `aggressive`: higher tool-call budget plus fallback and dynamic model selection
  - `custom`: do not apply preset defaults beyond the explicit field values you provide

#### `middleware.enabled`

- Type: `bool`
- Default: `true`
- Meaning: global switch for agent middleware assembly

#### `middleware.tool_call_limit_enabled`

- Type: `bool`
- Default: `true`
- Meaning: enables `ToolCallLimitMiddleware`

#### `middleware.tool_call_limit`

- Type: `int`
- Default: `6`
- Allowed range: `1` to `100`
- Meaning: per-run tool call ceiling for agents

#### `middleware.model_fallback_enabled`

- Type: `bool`
- Default: `false`
- Meaning: enables `ModelFallbackMiddleware`
- Note: only effective when `reasoning_model` is different from `chat_model`

#### `middleware.dynamic_model_selection_enabled`

- Type: `bool`
- Default: `false`
- Meaning: enables a custom `wrap_model_call` middleware that switches long inputs to `reasoning_model`
- Note: this is one of the LangChain 1.2-style middleware patterns now built into the template

#### `middleware.dynamic_model_selection_message_threshold`

- Type: `int`
- Default: `800`
- Allowed range: `50` to `20000`
- Meaning: approximate character threshold above which the middleware switches to `reasoning_model`

#### `middleware.pii`

- Type: object
- Meaning: config for `PIIMiddleware`

#### `middleware.pii.enabled`

- Type: `bool`
- Default: `false`

#### `middleware.pii.pii_types`

- Type: `list[str]`
- Default: `["email", "url"]`
- Meaning: PII types passed to `PIIMiddleware`

#### `middleware.pii.strategy`

- Type: `"block" | "redact" | "mask" | "hash"`
- Default: `"redact"`

#### `middleware.pii.apply_to_input`

- Type: `bool`
- Default: `true`

#### `middleware.pii.apply_to_output`

- Type: `bool`
- Default: `false`

#### `middleware.pii.apply_to_tool_results`

- Type: `bool`
- Default: `false`

#### `middleware.summarization`

- Type: object
- Meaning: config for `SummarizationMiddleware` on memory-enabled agents

#### `middleware.summarization.enabled`

- Type: `bool`
- Default: `true`

#### `middleware.summarization.trigger_messages`

- Type: `int`
- Default: `24`
- Allowed range: `4` to `200`
- Meaning: when conversation history reaches this message count, the middleware starts summarizing

#### `middleware.summarization.keep_messages`

- Type: `int`
- Default: `12`
- Allowed range: `2` to `100`
- Meaning: recent messages retained alongside the generated summary
- Validation: must be smaller than `trigger_messages`

## Provider section

Each provider entry under `providers` uses the same schema.

### `type`

- Type: `"openai_compatible" | "ollama"`
- Default: `"openai_compatible"`
- Enforcement: hard enforcement
- Meaning: provider driver type

### `enabled`

- Type: `bool`
- Default: `true`
- Enforcement: hard enforcement
- Meaning: whether this provider can be selected

### `api_key`

- Type: `str`
- Default: `""`
- Enforcement: hard enforcement for remote providers
- Meaning: API key for remote providers
- Notes:
  - required for `openai_compatible`
  - can be blank for `ollama`
  - placeholder values are treated as invalid by health checks

### `base_url`

- Type: `str`
- Default: `""`
- Enforcement: hard enforcement
- Meaning: provider API base URL
- Required:
  - yes for `openai_compatible`
  - typically `http://localhost:11434` for `ollama`

### `chat_model`

- Type: `str`
- Default: `""`
- Enforcement: hard enforcement for actual calls
- Meaning: primary chat model name
- Required: yes for `ollama`, practically yes for all providers

### `reasoning_model`

- Type: `str`
- Default: `""`
- Enforcement: default override
- Meaning: optional reasoning-oriented model name for tasks that need deeper reasoning
- Note: only affects code paths that explicitly choose a reasoning model

### `embedding_model`

- Type: `str`
- Default: `""`
- Enforcement: hard enforcement for embedding workflows
- Meaning: embedding model used for vector indexing and retrieval

### `rerank_model`

- Type: `str`
- Default: `""`
- Enforcement: hard enforcement for rerank workflows
- Meaning: optional rerank model name used in retrieval pipelines

### `embedding_dimensions`

- Type: `int | null`
- Default: `null` unless a provider default sets it
- Enforcement: mixed
- Meaning: expected vector dimension for the configured embedding model
- Behavior:
  - used in doctor output for visibility
  - appended to default collection names to avoid mixing embeddings from incompatible vector spaces
- Note: this does not resize embeddings; it is metadata used to keep collections isolated and self-describing

### `temperature`

- Type: `float`
- Default: `0.1`
- Enforcement: hard enforcement
- Meaning: model sampling temperature sent to provider clients
- Typical range: `0.0` to `1.0`

### `request_timeout`

- Type: `float`
- Default: `60.0` in code, repository config uses `20` for remote providers and `60` for ollama by default
- Constraint: must be greater than `0`
- Enforcement: hard enforcement
- Meaning: HTTP request timeout in seconds

### `max_retries`

- Type: `int`
- Default: `2` in code, repository config uses `1` for remote providers and `2` for ollama by default
- Allowed range: `0` to `10`
- Enforcement: hard enforcement
- Meaning: retry count for provider client requests

## Default provider entries

The repository ships with four named providers:

- `openai`: OpenAI-compatible API defaults
- `qwen`: DashScope-compatible defaults
- `deepseek`: DeepSeek-compatible defaults
- `ollama`: local Ollama defaults

Current repository defaults:

- `openai.embedding_model = text-embedding-3-small`, `embedding_dimensions = 1536`
- `qwen.embedding_model = text-embedding-v1`, `embedding_dimensions = 1536`
- `deepseek.embedding_model = ""`, `embedding_dimensions = null`
- `ollama.embedding_model = nomic-embed-text`, `embedding_dimensions = 768` in the example config

You can keep these names or add your own provider entries, as long as `runtime.active_provider` and `runtime.rerank_provider` point to valid provider names.

## Validation rules

The config loader currently validates these rules:

- top-level YAML value must be a mapping
- `openai_compatible` providers must define `base_url`
- `ollama` providers must define `chat_model`
- selected remote providers must not use placeholder API keys

## Example

```yaml
runtime:
  active_provider: ollama
  rerank_provider: ollama
  http_proxy: http://127.0.0.1:7890
  https_proxy: http://127.0.0.1:7890
  default_collection_name: demo_collection
  default_persist_directory: data/index/chroma
  top_k: 4
  chunk_size: 500
  chunk_overlap: 100
  hybrid_rrf_k: 60
  response_language: zh-CN
  response_format: markdown
  answer_style: balanced
  default_output_mode: concise
  log_level: INFO
  third_party_log_level: WARNING
  log_file: logs/lc_templates.log
  max_citations: 3
  routing_confidence_threshold: 0.55
  rag_no_answer_message: "I cannot answer confidently from the provided context."
  middleware:
    profile: balanced
    enabled: true
    tool_call_limit_enabled: true
    tool_call_limit: 6
    model_fallback_enabled: false
    dynamic_model_selection_enabled: false
    dynamic_model_selection_message_threshold: 800
    pii:
      enabled: false
      pii_types:
        - email
        - url
      strategy: redact
      apply_to_input: true
      apply_to_output: false
      apply_to_tool_results: false
    summarization:
      enabled: true
      trigger_messages: 24
      keep_messages: 12

providers:
  ollama:
    type: ollama
    enabled: true
    api_key: ""
    base_url: http://localhost:11434
    chat_model: qwen3:4b
    reasoning_model: qwen3:4b
    embedding_model: bge-m3
    embedding_dimensions: 1024
    rerank_model: ""
    temperature: 0.1
    request_timeout: 60
    max_retries: 2
```
