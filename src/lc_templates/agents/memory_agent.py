from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from lc_templates.core.hooks import emit_event
from lc_templates.core.logging import get_logger
from lc_templates.core.middleware import build_agent_middleware
from lc_templates.core.models import build_chat_model
from lc_templates.core.output import build_agent_execution_result
from lc_templates.core.prompts import build_agent_system_prompt
from lc_templates.core.schemas import ExecutionMetadata
from lc_templates.tools.common import COMMON_TOOLS

_MEMORY = InMemorySaver()
logger = get_logger(__name__)


def build_memory_agent():
    model = build_chat_model()
    middleware = build_agent_middleware(has_memory=True)
    logger.info("Building memory agent. tool_count=%s", len(COMMON_TOOLS))
    emit_event(
        "memory_agent.build",
        message="Building memory agent.",
        meta=ExecutionMetadata(operation="memory_agent"),
        payload={"tool_count": len(COMMON_TOOLS)},
    )
    return create_agent(
        model=model,
        tools=COMMON_TOOLS,
        middleware=middleware,
        checkpointer=_MEMORY,
        system_prompt=build_agent_system_prompt(has_memory=True),
    )


def run_memory_agent(thread_id: str, user_input: str):
    agent = build_memory_agent()
    logger.info("Running memory agent. thread_id=%s input_length=%s", thread_id, len(user_input))
    emit_event(
        "memory_agent.run",
        message="Running memory agent.",
        meta=ExecutionMetadata(operation="memory_agent"),
        payload={"thread_id": thread_id, "input_length": len(user_input)},
    )
    return agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": thread_id}},
    )


def run_memory_agent_result(thread_id: str, user_input: str):
    return build_agent_execution_result(run_memory_agent(thread_id, user_input))
