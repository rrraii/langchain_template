import _bootstrap as _bootstrap  # noqa: F401
from _args import build_example_parser, resolve_output_mode

from lc_templates import create_app
from lc_templates.core.output import to_pretty_json

if __name__ == "__main__":
    args = build_example_parser("Run the text task bundle example.").parse_args()
    output_mode = resolve_output_mode(args)
    app = create_app()
    text = "患者近一周反复头晕，既往有高血压病史，目前血压 160/100 mmHg。"

    if output_mode == "json":
        print(to_pretty_json(app.run_text_tasks(text)))
    else:
        print(app.run_text_tasks_display(text, verbose=output_mode == "verbose"))
