import _bootstrap as _bootstrap  # noqa: F401
from _args import build_example_parser, resolve_output_mode

from lc_templates import create_app
from lc_templates.core.output import to_pretty_json

if __name__ == "__main__":
    args = build_example_parser("Run the default agent example.").parse_args()
    output_mode = resolve_output_mode(args)
    app = create_app()
    text = "帮我计算 (15 + 27) * 3，并顺便告诉我现在时间。"

    if output_mode == "json":
        print(to_pretty_json(app.agent(text)))
    else:
        print(app.agent_display(text, verbose=output_mode == "verbose"))
