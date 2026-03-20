import _bootstrap as _bootstrap  # noqa: F401
from _args import build_example_parser, resolve_output_mode

from lc_templates import create_app
from lc_templates.core.output import to_pretty_json

if __name__ == "__main__":
    args = build_example_parser("Run the streaming chat example.").parse_args()
    output_mode = resolve_output_mode(args)
    app = create_app()
    prompt = "Please explain what RAG is and give one medical knowledge-base example."
    chunks = list(app.stream_chat(prompt))
    full_text = "".join(chunks)

    if output_mode == "json":
        print(to_pretty_json({"prompt": prompt, "chunks": chunks, "text": full_text}))
    else:
        if output_mode == "verbose":
            print("Streaming response:\n")
        print(full_text)
