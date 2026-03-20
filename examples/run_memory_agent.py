import _bootstrap as _bootstrap  # noqa: F401
from _args import build_example_parser, resolve_output_mode

from lc_templates import create_app
from lc_templates.core.output import to_pretty_json

if __name__ == "__main__":
    args = build_example_parser("Run the memory agent example.").parse_args()
    output_mode = resolve_output_mode(args)
    app = create_app()
    thread_id = "demo-thread"
    first_text = "我叫小王，我在学 LangChain。"
    second_text = "你还记得我叫什么吗？"

    if output_mode == "json":
        print(
            to_pretty_json(
                {
                    "turn_1": app.memory_agent(thread_id, first_text),
                    "turn_2": app.memory_agent(thread_id, second_text),
                }
            )
        )
    else:
        print("=== Turn 1 ===")
        print(app.memory_agent_display(thread_id, first_text, verbose=output_mode == "verbose"))

        print("\n=== Turn 2 ===")
        print(app.memory_agent_display(thread_id, second_text, verbose=output_mode == "verbose"))
