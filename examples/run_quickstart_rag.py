import _bootstrap as _bootstrap  # noqa: F401
from _args import build_example_parser, resolve_output_mode

from lc_templates import create_app
from lc_templates.core.config import get_resolved_config_path, scaffold_config
from lc_templates.core.output import to_pretty_json

if __name__ == "__main__":
    default_file_path = "examples/data/medical_demo.txt"
    default_question = "高血压患者平时需要注意什么？"

    parser = build_example_parser(
        "Run a quickstart RAG walkthrough: config scaffold, doctor, "
        "indexing, and optional live query."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also build the demo knowledge base and run a live RAG question.",
    )
    parser.add_argument(
        "--file-path",
        default=default_file_path,
        help="Document file used for the quickstart RAG example.",
    )
    parser.add_argument(
        "--question",
        default=default_question,
        help="Question to ask after indexing the demo file.",
    )
    args = parser.parse_args()
    output_mode = resolve_output_mode(args)

    config_path = get_resolved_config_path()
    config_exists = config_path.exists()
    if not config_exists:
        scaffold_config(str(config_path))

    app = create_app()
    doctor = app.doctor()

    if output_mode == "json":
        payload = {
            "config_path": str(config_path),
            "config_created": not config_exists,
            "doctor": doctor,
            "demo_file_path": args.file_path,
        }
        if args.live:
            payload["index"] = app.index_file(args.file_path)
            payload["rag"] = app.ask_rag_structured(args.question)
        print(to_pretty_json(payload))
    else:
        print("=== Quickstart RAG ===")
        print(f"Config path: {config_path}")
        print(f"Config created: {'yes' if not config_exists else 'no'}")
        print(f"Demo file: {args.file_path}")

        print("\n=== Doctor ===")
        print(app.doctor_display(verbose=output_mode == "verbose"))

        if args.live:
            print("\n=== Index ===")
            print(app.index_display(args.file_path, verbose=output_mode == "verbose"))

            print("\n=== RAG ===")
            if output_mode == "verbose":
                print(app.ask_rag_structured(args.question).model_dump_json(indent=2))
            else:
                print(app.ask_rag_rendered(args.question))
