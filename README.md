# langchain12-templates

[Chinese README](./README.zh-CN.md)

Engineering-ready LangChain templates for chat, agents, structured output, and RAG workflows.

This repository is designed to be easy to clone, configure, and extend for real projects. It provides:

- A provider-driven model factory for OpenAI-compatible APIs and Ollama
- A high-level `TemplateApp` facade for common application flows
- Chat, streaming, summarization, classification, extraction, routing, agent, memory-agent, and RAG templates
- Structured RAG with citation validation and no-answer fallback
- Configurable LangChain 1.2 agent middleware for tool limits, PII protection, memory summarization, model fallback, and dynamic model selection
- Middleware profiles: `safe`, `balanced`, `aggressive`, and `custom`
- Stable result schemas with `trace_id`, `latency_ms`, fallback metadata, normalized payloads, and hook events
- Event hooks for app-level observability and custom integrations
- CLI commands, examples, tests, docs, CI, and repository metadata for long-term maintenance

## Installation

```bash
pip install -r requirements.txt
```

For editable development installs:

```bash
pip install -e ".[dev]"
```

For users in mainland China:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Quick Start

### 1. Initialize config

```bash
lc-templates init-config
```

Or copy:

- `config/config.example.yaml` -> `config/config.yaml`

`api_key` supports both forms:

```yaml
api_key: DASHSCOPE_API_KEY
```

```yaml
api_key: ${DASHSCOPE_API_KEY}
```

### 2. Use the high-level facade

```python
from lc_templates import create_app

app = create_app()

print(app.chat("Explain what this template library is for."))
print(app.summarize("Long text goes here."))
print(app.route("Please summarize this meeting note."))
print(app.run("Please summarize this meeting note.").text)
print(app.agent("Calculate (15 + 27) * 3 and tell me the current time.").final_text)
print(app.doctor_display())
```

### 3. Build and query a knowledge base

```python
from lc_templates import create_app

app = create_app()
app.index_file("examples/data/medical_demo.txt")
print(app.ask_rag_rendered("What should patients with hypertension pay attention to?"))
```

### 4. Subscribe to events

```python
from lc_templates import create_app

app = create_app()
app.on_event(lambda event: print(event.name, event.trace_id, event.payload))
print(app.classify_label("Please summarize this note.", ["summarize", "chat"]))
```

## CLI Usage

```bash
lc-templates init-config
lc-templates chat "Explain what RAG is."
lc-templates summarize "Summarize this text."
lc-templates classify "Summarize this memo." --labels rag extract summarize chat
lc-templates route "Please summarize this meeting note."
lc-templates run "Please summarize this meeting note."
lc-templates run "What should patients with hypertension pay attention to?" --use-rag --collection-name demo_collection__qwen__text-embedding-v1__1536d
lc-templates agent "Calculate (15 + 27) * 3 and tell me the current time." --output json
lc-templates doctor --output verbose
lc-templates config --output json
lc-templates index examples/data/medical_demo.txt
lc-templates rag "What should patients with hypertension pay attention to?"
```

## Important Config

Frequently used runtime fields:

- `runtime.active_provider`
- `runtime.default_output_mode`
- `runtime.log_level`
- `runtime.third_party_log_level`
- `runtime.log_file`
- `runtime.default_collection_name`
- `runtime.default_persist_directory`
- `runtime.routing_confidence_threshold`
- `runtime.rag_no_answer_message`
- `runtime.middleware`

Frequently used provider fields:

- `providers.<name>.api_key`
- `providers.<name>.base_url`
- `providers.<name>.chat_model`
- `providers.<name>.reasoning_model`
- `providers.<name>.embedding_model`
- `providers.<name>.embedding_dimensions`
- `providers.<name>.request_timeout`
- `providers.<name>.max_retries`

Middleware fields:

- `runtime.middleware.profile`
- `runtime.middleware.tool_call_limit_enabled`
- `runtime.middleware.tool_call_limit`
- `runtime.middleware.model_fallback_enabled`
- `runtime.middleware.dynamic_model_selection_enabled`
- `runtime.middleware.dynamic_model_selection_message_threshold`
- `runtime.middleware.pii.*`
- `runtime.middleware.summarization.*`

See full parameter docs in:

- `docs/configuration.md`

## Recommended API

- `lc_templates.create_app`
- `lc_templates.TemplateApp`
- `lc_templates.register_event_hook`
- `lc_templates.unregister_event_hook`
- `lc_templates.clear_event_hooks`
- `TemplateApp.run`
- `TemplateApp.run_display`
- `TemplateApp.agent`
- `TemplateApp.memory_agent`
- `TemplateApp.ask_rag_structured`
- `TemplateApp.ask_rag_rendered`
- `TemplateApp.index_file`
- `TemplateApp.doctor`
- `TemplateApp.doctor_recommendations`
- `TemplateApp.doctor_recommended_profile`
- `TemplateApp.init_config`
- `TemplateApp.on_event`

## Result Schemas

Common schemas:

- `ResultEnvelope`
- `HookEvent`
- `WorkflowExecutionResult`
- `AgentExecutionResult`
- `ClassificationResult`
- `ExtractionResult`
- `GroundedAnswer`
- `KnowledgeBaseBuildResult`
- `RouteDecision`
- `TaskBundleResult`

`ResultEnvelope`-based results include:

- `trace_id`
- `latency_ms`
- `status`
- `fallback_used`
- `error_reason`
- `meta`

## Logging and Diagnostics

Recommended defaults:

```yaml
runtime:
  log_level: INFO
  third_party_log_level: WARNING
  log_file: logs/lc_templates.log
```

Notes:

- Framework logs go to the console and optionally to a file
- Noisy third-party SDK logs are lowered by `third_party_log_level`
- `doctor()` returns both `warnings` and `recommendations`
- `doctor()` also returns `recommended_middleware_profile`
- `doctor_display(verbose=True)` is the fastest local debugging view
- Event hooks provide a code-level observability path in addition to log files

## Project Structure

```text
langchain12-templates/
  .github/
  config/
  docs/
  examples/
  src/lc_templates/
    agents/
    chains/
    core/
    rag/
    tools/
    workflows/
    app.py
    cli.py
    __main__.py
  tests/
  README.md
  README.zh-CN.md
  pyproject.toml
```

## Examples

Available in `examples/`:

- `run_app.py`
- `run_basic_chat.py`
- `run_agent.py`
- `run_memory_agent.py`
- `run_structured_output.py`
- `run_rag.py`
- `run_router.py`
- `run_streaming.py`
- `run_tasks.py`
- `run_hybrid_search.py`
- `run_quickstart.py`
- `run_quickstart_rag.py`

## Testing

```bash
pytest -q
```

Coverage:

```bash
pytest --cov=lc_templates --cov-report=term-missing
```

## Design Goals

- Reduce first-run setup friction
- Improve weak-model stability with prompt constraints and schema validation
- Expose clean top-level APIs for application teams
- Preserve raw outputs for debugging while returning normalized results for production use
- Keep config, samples, CLI, docs, and tests aligned over time
