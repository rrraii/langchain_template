import httpx
from langchain.agents import create_agent
from openai import APIConnectionError

from lc_templates.core.models import build_chat_model, get_active_provider_name
from lc_templates.core.output import build_agent_execution_result
from lc_templates.core.prompts import build_agent_system_prompt
from lc_templates.tools.common import COMMON_TOOLS


def build_basic_agent():
    model = build_chat_model()
    return create_agent(
        model=model,
        tools=COMMON_TOOLS,
        system_prompt=build_agent_system_prompt(),
    )


def run_basic_agent(user_input: str):
    agent = build_basic_agent()
    try:
        return agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    except APIConnectionError as exc:
        provider_name = get_active_provider_name()
        raise RuntimeError(
            "Agent invoked successfully, but the model provider connection failed. "
            f"Current provider: {provider_name}. "
            "Please check base_url, proxy settings, TLS/certificate interception, "
            "and whether the provider service is reachable."
        ) from exc
    except httpx.ConnectError as exc:
        provider_name = get_active_provider_name()
        raise RuntimeError(
            "Agent invoked successfully, but the upstream model endpoint could not be reached. "
            f"Current provider: {provider_name}. "
            "This usually means a proxy, TLS handshake, or network access issue "
            "rather than a bad messages payload."
        ) from exc


def run_basic_agent_result(user_input: str):
    return build_agent_execution_result(run_basic_agent(user_input))
