import _bootstrap as _bootstrap  # noqa: F401
from _args import build_example_parser, resolve_output_mode

from lc_templates import create_app
from lc_templates.core.output import to_pretty_json

if __name__ == "__main__":
    args = build_example_parser("Run the high-level app example.").parse_args()
    output_mode = resolve_output_mode(args)
    app = create_app()

    chat_question = "请用两句话说明这个模板库适合做什么。"
    classify_input = "请总结这段会议记录。"
    classify_labels = ["rag", "extract", "summarize", "chat"]
    agent_input = "帮我计算 (15 + 27) * 3，并告诉我现在时间。"

    if output_mode == "json":
        print(
            to_pretty_json(
                {
                    "chat": {"text": app.chat(chat_question)},
                    "classification": app.classify(classify_input, classify_labels),
                    "agent": app.agent(agent_input),
                }
            )
        )
    else:
        print("=== Chat ===")
        print(app.chat(chat_question))

        print("\n=== Classify ===")
        print(
            app.classify_text_result(
                classify_input,
                classify_labels,
                verbose=output_mode == "verbose",
            )
        )

        print("\n=== Agent ===")
        print(app.agent_display(agent_input, verbose=output_mode == "verbose"))
