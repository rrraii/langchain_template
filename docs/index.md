# langchain12-templates

`langchain12-templates` is an engineering-oriented template repository for building LangChain applications with better structure, safer defaults, and clearer extension points.

## What it includes

- Provider-driven model factory
- High-level `TemplateApp` facade
- Chat, agent, structured-output, and RAG templates
- CLI and health-check commands
- Config-first runtime behavior
- Tests and repository automation

## Recommended starting point

```python
from lc_templates import create_app

app = create_app()
print(app.chat("Explain what this template library is for."))
```

See the root [README](../README.md) for installation instructions.
