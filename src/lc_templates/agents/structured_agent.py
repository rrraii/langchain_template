from langchain.agents import create_agent

from lc_templates.core.models import build_chat_model
from lc_templates.core.prompts import build_agent_system_prompt
from lc_templates.core.schemas import CitationAnswer
from lc_templates.tools.common import COMMON_TOOLS


def build_structured_agent():
    model = build_chat_model()
    return create_agent(
        model=model,
        tools=COMMON_TOOLS,
        response_format=CitationAnswer,
        system_prompt=build_agent_system_prompt(structured=True),
    )
