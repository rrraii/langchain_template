# langchain12-templates

[中文文档](./README.zh-CN.md)

Engineering-ready LangChain templates for chat, agents, structured output, and RAG workflows.

This repository is designed to be easy to clone, configure, and extend for real projects. It provides:

- A provider-driven model factory for OpenAI-compatible APIs and Ollama
- A high-level `TemplateApp` facade for common application flows
- Chat, streaming, summarization, classification, extraction, routing, and agent templates
- Structured RAG with citation validation and no-answer fallback
- Configurable response style, response format, routing threshold, chunking, retries, and timeouts
- Standard-library and `pytest` test support without requiring live model calls

## Why This Template Is Stronger Than a Typical Demo Repo

- High-level facade and CLI for direct application use
- Health checks that surface config and provider readiness early
- Structured outputs with fallback parsing for weaker models
- RAG grounding checks with citation filtering and no-answer fallback
- Packaging, docs, tests, and repository metadata aligned for long-term maintenance

## Highlights

- Standard package imports: use `lc_templates`
- Config-first runtime behavior in `config/config.yaml`
- Stable result wrappers for agent execution, provider health, and knowledge-base indexing
- CLI entrypoint for common tasks and configuration inspection
- GitHub-friendly repository layout with CI, issue templates, PR template, and security policy

## Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes both runtime dependencies and common repository tooling such as `pytest`, `ruff`, `build`, and `twine`.

If you prefer editable installation with optional development extras:

```bash
pip install -e ".[dev]"
```

For users in mainland China, you can use a mirror:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Quick Start

### 1. Configure a provider

Copy `config/config.example.yaml` to `config/config.yaml` and update the provider settings.

Environment variables are supported in both of these forms:

```yaml
api_key: DASHSCOPE_API_KEY
```

```yaml
api_key: ${DASHSCOPE_API_KEY}
```

### 2. Use the high-level app facade

```python
from lc_templates import create_app

app = create_app()

print(app.chat("Explain what this template library is for."))
print(app.summarize("Long text goes here."))
print(app.route("Please summarize this meeting note."))
print(app.agent("Calculate (15 + 27) * 3 and tell me the current time.").final_text)
print(app.version())
print(app.config()["runtime"]["active_provider"])
print(app.doctor().model_dump())
```

### 3. Build and query a knowledge base

```python
from lc_templates import create_app

app = create_app()
app.index_file("examples/data/medical_demo.txt")

answer = app.ask_rag_rendered("What should patients with hypertension pay attention to?")
print(answer)

one_shot = app.ask_rag_from_file(
    "examples/data/medical_demo.txt",
    "What should patients with hypertension pay attention to?",
)
print(one_shot)
```

## CLI Usage

After installation, you can use the packaged CLI:

```bash
lc-templates chat "Explain what RAG is."
lc-templates summarize "Summarize this text."
lc-templates classify "Summarize this memo." --labels rag extract summarize chat
lc-templates route "Please summarize this meeting note."
lc-templates agent "Calculate (15 + 27) * 3 and tell me the current time." --output json
lc-templates classify "Summarize this memo." --labels rag extract summarize chat --output verbose
lc-templates version
lc-templates config
lc-templates doctor
lc-templates index examples/data/medical_demo.txt
lc-templates rag "What should patients with hypertension pay attention to?"
lc-templates rag "What should patients with hypertension pay attention to?" --output json
```

You can also run it as a module:

```bash
python -m lc_templates chat "Hello"
```

## Configuration

The main runtime configuration lives in `config/config.yaml`.

Important runtime fields:

- `active_provider`
- `rerank_provider`
- `http_proxy`
- `https_proxy`
- `default_collection_name`
- `default_persist_directory`
  Relative paths resolve from the project root.
- `top_k`
- `chunk_size`
- `chunk_overlap`
- `hybrid_rrf_k`
- `response_language`
- `response_format`
- `answer_style`
- `default_output_mode`
- `log_level`
- `third_party_log_level`
- `log_file`
  Relative paths resolve from the project root.
- `max_citations`
- `routing_confidence_threshold`
- `rag_no_answer_message`

Important provider fields:

- `base_url`
- `api_key`
- `chat_model`
- `reasoning_model`
- `embedding_model`
- `embedding_dimensions`
- `rerank_model`
- `temperature`
- `request_timeout`
- `max_retries`

For example scripts, output mode can be controlled in two ways:

- Set `runtime.default_output_mode` in `config/config.yaml`
- Override it per run with `--output concise`, `--output verbose`, or `--output json`

The CLI follows the same output convention for supported commands.

## Project Structure

```text
langchain12-templates/
├─ .github/
├─ config/
├─ examples/
├─ src/lc_templates/
│  ├─ agents/
│  ├─ chains/
│  ├─ core/
│  ├─ rag/
│  ├─ tools/
│  ├─ workflows/
│  ├─ app.py
│  ├─ cli.py
│  └─ __main__.py
├─ tests/
├─ CHANGELOG.md
├─ CONTRIBUTING.md
├─ SECURITY.md
├─ LICENSE
├─ Makefile
├─ pyproject.toml
├─ README.md
└─ README.zh-CN.md
```

## Main APIs

Recommended entrypoint:

- `lc_templates.create_app`
- `lc_templates.TemplateApp`

Useful high-level facade methods:

- `app.chat`
- `app.version`
- `app.config`
- `app.summarize`
- `app.classify`
- `app.extract`
- `app.route`
- `app.agent`
- `app.memory_agent`
- `app.index_file`
- `app.ask_rag_rendered`
- `app.ask_rag_structured`
- `app.ask_rag_from_file`
- `app.doctor`

Useful lower-level modules:

- `lc_templates.chains.basic_chat`
- `lc_templates.chains.structured_output`
- `lc_templates.agents.basic_agent`
- `lc_templates.agents.memory_agent`
- `lc_templates.rag.pipeline`

## Testing

Run with `pytest`:

```bash
pytest -q
```

Run with coverage:

```bash
pytest --cov=lc_templates --cov-report=term-missing
```

Run pre-commit hooks locally:

```bash
pre-commit run --all-files
```

Or with the standard library:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Repository Standards

This repository includes:

- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- GitHub issue templates
- GitHub pull request template
- GitHub Actions CI workflow
- `.editorconfig`
- `.pre-commit-config.yaml`
- MkDocs documentation skeleton in `docs/`

## Examples

Example scripts are available in `examples/`:

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

## Design Goals

- Make model behavior more stable through prompt constraints and schema validation
- Keep configuration centralized and easy to override
- Expose clean top-level APIs for downstream application teams
- Preserve raw results for debugging while returning normalized objects for production use
- Provide enough tests, examples, and repository metadata to support long-term maintenance
