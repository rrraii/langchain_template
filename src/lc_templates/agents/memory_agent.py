from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from lc_templates.core.models import build_chat_model
from lc_templates.core.output import build_agent_execution_result
from lc_templates.core.prompts import build_agent_system_prompt
from lc_templates.tools.common import COMMON_TOOLS

_MEMORY = InMemorySaver()


def build_memory_agent():
    model = build_chat_model()
    return create_agent(
        model=model,
        tools=COMMON_TOOLS,
        checkpointer=_MEMORY,
        system_prompt=build_agent_system_prompt(has_memory=True),
    )


def run_memory_agent(thread_id: str, user_input: str):
    agent = build_memory_agent()
    return agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": thread_id}},
    )


def run_memory_agent_result(thread_id: str, user_input: str):
    return build_agent_execution_result(run_memory_agent(thread_id, user_input))
