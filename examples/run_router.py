import _bootstrap as _bootstrap  # noqa: F401
from _args import build_example_parser, resolve_output_mode

from lc_templates import create_app
from lc_templates.core.output import to_pretty_json

if __name__ == "__main__":
    args = build_example_parser("Run the routing example.").parse_args()
    output_mode = resolve_output_mode(args)
    app = create_app()
    text = "Please summarize the key points from this meeting note."

    if output_mode == "json":
        print(to_pretty_json({"route": app.route(text), "name": app.route_name(text)}))
    else:
        print(app.route_display(text, verbose=output_mode == "verbose"))
