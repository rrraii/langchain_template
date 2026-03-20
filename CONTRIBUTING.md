# Contributing

Thanks for contributing to `langchain12-templates`.

## Local Setup

```bash
pip install -r requirements.txt
```

Or for editable development mode:

```bash
pip install -e ".[dev]"
```

## Running Tests

Standard library test command:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

If you install `pytest`:

```bash
pytest -q
```

## Style

- Prefer small, composable functions
- Keep configuration centralized in `config/config.yaml`
- Prefer high-level APIs through `TemplateApp` when adding examples
- Add tests for new behavior whenever possible
- Avoid breaking the package import path `lc_templates`

## Pull Requests

Please include:

- A short problem statement
- A summary of the change
- Notes on any config, behavior, or compatibility impact
- Tests or validation steps
