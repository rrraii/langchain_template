import _bootstrap as _bootstrap  # noqa: F401
from _args import build_example_parser, resolve_output_mode

from lc_templates import create_app
from lc_templates.core.output import to_pretty_json

if __name__ == "__main__":
    args = build_example_parser("Run the basic chat example.").parse_args()
    output_mode = resolve_output_mode(args)
    app = create_app()
    text = app.chat("请用 3 句话解释什么是 LangChain 1.2。")

    if output_mode == "json":
        print(to_pretty_json({"text": text}))
    else:
        print(text)
