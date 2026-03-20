from collections.abc import Iterable

from lc_templates.core.models import build_chat_model
from lc_templates.core.output import coerce_response_text, extract_text_content
from lc_templates.core.prompts import build_general_chat_system_prompt


def stream_chat(user_input: str) -> Iterable[str]:
    model = build_chat_model()
    for chunk in model.stream(
        [
            ("system", build_general_chat_system_prompt()),
            ("human", user_input),
        ]
    ):
        text = extract_text_content(chunk)
        if text:
            yield text


def batch_chat(prompts: list[str]) -> list[str]:
    model = build_chat_model()
    messages = [
        [("system", build_general_chat_system_prompt()), ("human", prompt)] for prompt in prompts
    ]
    responses = model.batch(messages)
    return [coerce_response_text(response) for response in responses]
