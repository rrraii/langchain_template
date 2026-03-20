from __future__ import annotations

import argparse
import sys

from lc_templates import __version__, create_app
from lc_templates.core.output import to_pretty_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CLI for lc_templates. Supported commands can use --output "
            "concise|verbose|json to control display style."
        )
    )
    parser.add_argument("--config", help="Optional path to config YAML.", default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)

    chat = subparsers.add_parser("chat", help="Run a basic chat request.")
    chat.add_argument("text")
    chat.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

    summarize = subparsers.add_parser("summarize", help="Summarize text.")
    summarize.add_argument("text")
    summarize.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

    classify = subparsers.add_parser("classify", help="Classify text into given labels.")
    classify.add_argument("text")
    classify.add_argument("--labels", nargs="+", required=True)
    classify.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

    route = subparsers.add_parser("route", help="Route text to a workflow.")
    route.add_argument("text")
    route.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

    subparsers.add_parser("doctor", help="Inspect configuration and provider readiness.")
    subparsers.add_parser("version", help="Print the installed package version.")
    subparsers.add_parser("config", help="Print the effective runtime configuration.")

    agent = subparsers.add_parser("agent", help="Run the default tool-using agent.")
    agent.add_argument("text")
    agent.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

    memory = subparsers.add_parser("memory-agent", help="Run the memory agent.")
    memory.add_argument("thread_id")
    memory.add_argument("text")
    memory.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

    rag = subparsers.add_parser("rag", help="Run RAG against an indexed collection.")
    rag.add_argument("question")
    rag.add_argument("--persist-directory", default=None)
    rag.add_argument("--collection-name", default=None)
    rag.add_argument("--k", type=int, default=None)
    rag.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

    index_cmd = subparsers.add_parser("index", help="Build a vector store from a file.")
    index_cmd.add_argument("path")
    index_cmd.add_argument("--persist-directory", default=None)
    index_cmd.add_argument("--collection-name", default=None)
    index_cmd.add_argument("--chunk-size", type=int, default=None)
    index_cmd.add_argument("--chunk-overlap", type=int, default=None)
    index_cmd.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

    return parser


def _resolve_output_mode(app, args: argparse.Namespace) -> str:
    runtime = getattr(app.settings, "runtime", None)
    if runtime is not None:
        default_output_mode = runtime.default_output_mode
    else:
        settings_dump = app.settings.model_dump()
        default_output_mode = settings_dump["runtime"]["default_output_mode"]
    return getattr(args, "output", None) or default_output_mode


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    app = create_app(config_path=args.config)

    if args.command == "chat":
        text = app.chat(args.text)
        output_mode = _resolve_output_mode(app, args)
        print(to_pretty_json({"text": text}) if output_mode == "json" else text)
        return 0
    if args.command == "summarize":
        text = app.summarize(args.text)
        output_mode = _resolve_output_mode(app, args)
        print(to_pretty_json({"text": text}) if output_mode == "json" else text)
        return 0
    if args.command == "classify":
        output_mode = _resolve_output_mode(app, args)
        if output_mode == "json":
            print(to_pretty_json(app.classify(args.text, args.labels)))
        else:
            print(
                app.classify_text_result(
                    args.text,
                    args.labels,
                    verbose=output_mode == "verbose",
                )
            )
        return 0
    if args.command == "route":
        output_mode = _resolve_output_mode(app, args)
        if output_mode == "json":
            print(
                to_pretty_json(
                    {"route": app.route(args.text), "name": app.route_name(args.text)}
                )
            )
        else:
            print(app.route_display(args.text, verbose=output_mode == "verbose"))
        return 0
    if args.command == "doctor":
        print(to_pretty_json(app.doctor()))
        return 0
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "config":
        print(to_pretty_json(app.config()))
        return 0
    if args.command == "agent":
        output_mode = _resolve_output_mode(app, args)
        if output_mode == "json":
            print(to_pretty_json(app.agent(args.text)))
        else:
            print(app.agent_display(args.text, verbose=output_mode == "verbose"))
        return 0
    if args.command == "memory-agent":
        output_mode = _resolve_output_mode(app, args)
        if output_mode == "json":
            print(to_pretty_json(app.memory_agent(args.thread_id, args.text)))
        else:
            print(
                app.memory_agent_display(
                    args.thread_id,
                    args.text,
                    verbose=output_mode == "verbose",
                )
            )
        return 0
    if args.command == "rag":
        output_mode = _resolve_output_mode(app, args)
        if output_mode in {"json", "verbose"}:
            result = app.ask_rag_structured(
                args.question,
                persist_directory=args.persist_directory,
                collection_name=args.collection_name,
                k=args.k,
            )
            if output_mode == "json":
                print(to_pretty_json(result))
            else:
                print(result.model_dump_json(indent=2))
        else:
            print(
                app.ask_rag_rendered(
                    args.question,
                    persist_directory=args.persist_directory,
                    collection_name=args.collection_name,
                    k=args.k,
                )
            )
        return 0
    if args.command == "index":
        result = app.index_file(
            args.path,
            persist_directory=args.persist_directory,
            collection_name=args.collection_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        output_mode = _resolve_output_mode(app, args)
        if output_mode == "json":
            print(to_pretty_json(result))
        elif output_mode == "verbose":
            print(
                "\n".join(
                    [
                        f"Source: {result.source_path}",
                        f"Persist directory: {result.persist_directory}",
                        f"Collection: {result.collection_name}",
                        f"Documents: {result.document_count}",
                        f"Chunks: {result.chunk_count}",
                    ]
                )
            )
        else:
            print(
                f"Indexed {result.document_count} document(s) into "
                f"{result.collection_name} with {result.chunk_count} chunk(s)."
            )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
