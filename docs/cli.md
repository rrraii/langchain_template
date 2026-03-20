# CLI

The project exposes a console script:

```bash
lc-templates --help
```

Useful commands:

```bash
lc-templates doctor
lc-templates chat "Explain RAG."
lc-templates summarize "Long text..."
lc-templates classify "Summarize this memo." --labels rag extract summarize chat
lc-templates classify "Summarize this memo." --labels rag extract summarize chat --output verbose
lc-templates agent "Calculate (15 + 27) * 3 and tell me the current time." --output json
lc-templates index examples/data/medical_demo.txt
lc-templates rag "What should patients with hypertension pay attention to?"
lc-templates rag "What should patients with hypertension pay attention to?" --output json
```

Supported commands use a unified output mode:

- `--output concise`
- `--output verbose`
- `--output json`
