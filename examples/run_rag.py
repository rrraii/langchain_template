from pathlib import Path

import _bootstrap as _bootstrap  # noqa: F401
from _args import build_example_parser, resolve_output_mode

from lc_templates import create_app
from lc_templates.core.output import to_pretty_json


def _ensure_sample_file() -> str:
    sample_path = Path("data/medical_demo.txt")
    if not sample_path.exists():
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(
            "高血压患者需要长期监测血压，并注意低盐饮食。\n"
            "如果出现胸闷、头痛、头晕等症状，应及时就医。\n",
            encoding="utf-8",
        )
    return str(sample_path)


if __name__ == "__main__":
    args = build_example_parser("Run the RAG example.").parse_args()
    output_mode = resolve_output_mode(args)
    app = create_app()
    file_path = _ensure_sample_file()
    question = "高血压患者平时需要注意什么？"

    app.index_file(file_path)

    if output_mode == "json":
        print(to_pretty_json(app.ask_rag_structured(question)))
    elif output_mode == "verbose":
        print(app.ask_rag_structured(question).model_dump_json(indent=2))
    else:
        print(app.ask_rag_rendered(question))
