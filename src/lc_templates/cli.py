from __future__ import annotations

import argparse
import sys

from lc_templates import __version__, create_app
from lc_templates.core.config import scaffold_config
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

    extract = subparsers.add_parser("extract", help="Extract entities from text.")
    extract.add_argument("text")
    extract.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

    classify = subparsers.add_parser("classify", help="Classify text into given labels.")
    classify.add_argument("text")
    classify.add_argument("--labels", nargs="+", required=True)
    classify.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

    route = subparsers.add_parser("route", help="Route text to a workflow.")
    route.add_argument("text")
    route.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

    tasks = subparsers.add_parser("tasks", help="Run the bundled text tasks.")
    tasks.add_argument("text")
    tasks.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

    run_cmd = subparsers.add_parser(
        "run",
        help="Route text and execute the matched local workflow.",
    )
    run_cmd.add_argument("text")
    run_cmd.add_argument("--use-rag", action="store_true")
    run_cmd.add_argument("--persist-directory", default=None)
    run_cmd.add_argument("--collection-name", default=None)
    run_cmd.add_argument("--k", type=int, default=None)
    run_cmd.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

    doctor = subparsers.add_parser("doctor", help="Inspect configuration and provider readiness.")
    doctor.add_argument("--output", choices=["concise", "verbose", "json"], default=None)
    subparsers.add_parser("version", help="Print the installed package version.")
    config = subparsers.add_parser("config", help="Print the effective runtime configuration.")
    config.add_argument("--output", choices=["concise", "verbose", "json"], default=None)
    init_config = subparsers.add_parser(
        "init-config",
        help="Write config/config.yaml from config/config.example.yaml.",
    )
    init_config.add_argument("--path", default=None)
    init_config.add_argument("--force", action="store_true")
    init_config.add_argument("--output", choices=["concise", "verbose", "json"], default=None)

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

    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "init-config":
        output_mode = getattr(args, "output", None) or "concise"
        path = str(scaffold_config(args.path, overwrite=args.force))
        if output_mode == "json":
            print(to_pretty_json({"path": path, "overwritten": args.force}))
        else:
            print(f"Wrote config template to {path}")
        return 0

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
    if args.command == "extract":
        output_mode = _resolve_output_mode(app, args)
        if output_mode == "json":
            print(to_pretty_json(app.extract(args.text)))
        else:
            print(app.extract_display(args.text, verbose=output_mode == "verbose"))
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
            print(to_pretty_json(app.route_decision(args.text)))
        else:
            print(app.route_display(args.text, verbose=output_mode == "verbose"))
        return 0
    if args.command == "tasks":
        output_mode = _resolve_output_mode(app, args)
        if output_mode == "json":
            print(to_pretty_json(app.run_text_tasks(args.text)))
        else:
            print(app.run_text_tasks_display(args.text, verbose=output_mode == "verbose"))
        return 0
    if args.command == "run":
        output_mode = _resolve_output_mode(app, args)
        if output_mode == "json":
            print(
                to_pretty_json(
                    app.run(
                        args.text,
                        use_rag=args.use_rag,
                        persist_directory=args.persist_directory,
                        collection_name=args.collection_name,
                        k=args.k,
                    )
                )
            )
        else:
            print(
                app.run_display(
                    args.text,
                    verbose=output_mode == "verbose",
                    use_rag=args.use_rag,
                    persist_directory=args.persist_directory,
                    collection_name=args.collection_name,
                    k=args.k,
                )
            )
        return 0
    if args.command == "doctor":
        output_mode = _resolve_output_mode(app, args)
        if output_mode == "json":
            print(to_pretty_json(app.doctor()))
        else:
            print(app.doctor_display(verbose=output_mode == "verbose"))
        return 0
    if args.command == "config":
        output_mode = _resolve_output_mode(app, args)
        if output_mode == "json":
            print(to_pretty_json(app.config()))
        elif output_mode == "verbose":
            print(to_pretty_json(app.config()))
        else:
            runtime = app.config().get("runtime", {})
            print(
                "\n".join(
                    [
                        f"active_provider: {runtime.get('active_provider', '')}",
                        f"default_output_mode: {runtime.get('default_output_mode', '')}",
                    ]
                )
            )
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
        output_mode = _resolve_output_mode(app, args)
        if output_mode == "json":
            print(
                to_pretty_json(
                    app.index_file(
                        args.path,
                        persist_directory=args.persist_directory,
                        collection_name=args.collection_name,
                        chunk_size=args.chunk_size,
                        chunk_overlap=args.chunk_overlap,
                    )
                )
            )
        else:
            print(
                app.index_display(
                    args.path,
                    persist_directory=args.persist_directory,
                    collection_name=args.collection_name,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    verbose=output_mode == "verbose",
                )
            )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
