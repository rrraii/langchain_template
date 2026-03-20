from langchain_core.output_parsers import StrOutputParser

from lc_templates.core.models import build_chat_model
from lc_templates.core.output import normalize_text
from lc_templates.core.prompts import build_summarize_prompt


def summarize_text(text: str) -> str:
    prompt = build_summarize_prompt()
    model = build_chat_model()
    chain = prompt | model | StrOutputParser() | normalize_text
    return chain.invoke({"text": text})
