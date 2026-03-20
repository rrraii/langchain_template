import _bootstrap as _bootstrap  # noqa: F401
from _args import build_example_parser, resolve_output_mode

from lc_templates import create_app
from lc_templates.core.output import to_pretty_json

if __name__ == "__main__":
    args = build_example_parser("Run the structured output example.").parse_args()
    output_mode = resolve_output_mode(args)
    app = create_app()
    text = "患者 3 天前开始发热、咳嗽，今天体温 38.5 度，考虑上呼吸道感染。"
    labels = ["medical", "legal", "customer_service", "general"]

    if output_mode == "json":
        print(
            to_pretty_json(
                {
                    "classification": app.classify(text, labels),
                    "extraction": app.extract(text),
                }
            )
        )
    else:
        print("=== Classify ===")
        print(app.classify_text_result(text, labels, verbose=output_mode == "verbose"))

        print("\n=== Extract ===")
        print(app.extract_display(text, verbose=output_mode == "verbose"))
