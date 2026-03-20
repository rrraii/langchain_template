# Security Policy

## Supported Versions

The latest version on the default branch is the supported version.

## Reporting a Vulnerability

If you discover a security issue, please avoid opening a public issue with full exploit details.

Instead, share:

- A clear description of the issue
- Affected components or files
- Reproduction steps
- Potential impact

If public disclosure is unavoidable, keep the initial report minimal and avoid including secrets, tokens, or production credentials.

## Security Notes

- Do not commit real API keys into `config/config.yaml`
- Prefer environment-variable based secrets
- Review tool execution and model provider settings before exposing this template to untrusted users
- The provided `calculator` tool is intentionally restricted to arithmetic expressions
