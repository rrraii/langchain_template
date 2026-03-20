from __future__ import annotations

import argparse

from lc_templates.core.config import get_settings


def build_example_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output",
        choices=["concise", "verbose", "json"],
        default=None,
        help="Select the output mode. Defaults to runtime.default_output_mode from config.",
    )
    return parser


def resolve_output_mode(args: argparse.Namespace) -> str:
    return args.output or get_settings().runtime.default_output_mode
