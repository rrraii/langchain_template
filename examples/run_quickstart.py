import _bootstrap as _bootstrap  # noqa: F401
from _args import build_example_parser, resolve_output_mode

from lc_templates import create_app
from lc_templates.core.config import get_resolved_config_path, scaffold_config
from lc_templates.core.output import to_pretty_json

if __name__ == "__main__":
    live_prompt = "Please summarize this meeting note."
    parser = build_example_parser(
        "Run a quickstart walkthrough: config scaffold, doctor, and optional live run."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also run a live routed request after the quickstart checks.",
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
        }
        if args.live:
            payload["run"] = app.run(live_prompt)
        print(to_pretty_json(payload))
    else:
        print("=== Quickstart ===")
        print(f"Config path: {config_path}")
        print(f"Config created: {'yes' if not config_exists else 'no'}")

        print("\n=== Doctor ===")
        print(app.doctor_display(verbose=output_mode == "verbose"))

        if args.live:
            print("\n=== Live Run ===")
            print(
                app.run_display(
                    live_prompt,
                    verbose=output_mode == "verbose",
                )
            )
