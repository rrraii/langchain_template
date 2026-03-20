# Architecture

The repository is organized around a few stable layers:

- `core/`: configuration, prompts, schemas, output normalization, logging
- `chains/`: direct prompt-and-model pipelines
- `agents/`: tool-using and memory-enabled agent templates
- `rag/`: indexing, retrieval, ranking, and grounded answer generation
- `app.py`: high-level facade for downstream applications
- `cli.py`: command-line interface

## Design principles

- Prefer configuration over hard-coded runtime behavior
- Provide normalized outputs in addition to raw data
- Add schema validation and fallback strategies for weaker models
- Keep retrieval answers grounded through citation validation
