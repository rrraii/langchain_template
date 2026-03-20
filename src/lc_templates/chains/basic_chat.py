from lc_templates.core.models import build_chat_model
from lc_templates.core.output import coerce_response_text
from lc_templates.core.prompts import build_general_chat_system_prompt


def basic_chat(user_input: str) -> str:
    model = build_chat_model()
    response = model.invoke(
        [
            ("system", build_general_chat_system_prompt()),
            ("human", user_input),
        ]
    )
    return coerce_response_text(response)
