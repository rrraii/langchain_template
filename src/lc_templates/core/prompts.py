from langchain_core.prompts import ChatPromptTemplate

from lc_templates.core.config import get_settings
from lc_templates.core.output import get_response_format_instructions


def _base_assistant_rules() -> str:
    settings = get_settings().runtime
    return "\n".join(
        [
            "You are a reliable AI assistant with strong engineering discipline.",
            get_response_format_instructions(),
            "Prefer accurate, direct answers over speculation.",
            "State uncertainty clearly when the available information is insufficient.",
            "When answering from retrieval context, "
            f"use at most {settings.max_citations} short references.",
        ]
    )


def build_general_chat_system_prompt() -> str:
    return "\n".join(
        [
            _base_assistant_rules(),
            "When the user refers to 'this template library', 'this framework', or "
            "'this project' without more context, treat it as the current lc_templates project.",
            "Prefer practical, project-relevant answers over generic definitions.",
            "If the request is broad, answer with the most likely intent first instead of asking "
            "for unnecessary clarification.",
            "Lead with the answer, then add the minimum supporting detail needed.",
        ]
    )


def build_agent_system_prompt(*, has_memory: bool = False, structured: bool = False) -> str:
    instructions = [
        _base_assistant_rules(),
        "Use tools when a calculation, lookup, or external action is needed.",
        "Do not claim you lack a capability if an available tool can help.",
    ]
    if has_memory:
        instructions.append(
            "Use the conversation history in the same thread_id when it is relevant."
        )
    if structured:
        instructions.append("Return data that fully satisfies the required response schema.")
    return "\n".join(instructions)


def build_qa_prompt() -> ChatPromptTemplate:
    no_answer_message = get_settings().runtime.rag_no_answer_message
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "\n".join(
                    [
                        _base_assistant_rules(),
                        "Answer only from the provided context.",
                        f"If the context is insufficient, reply with exactly: {no_answer_message}",
                    ]
                ),
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}\n\n"
                "Answer using only the context and keep references short when relevant.",
            ),
        ]
    )


def build_summarize_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "\n".join(
                    [
                        _base_assistant_rules(),
                        "Summarize faithfully without adding unsupported facts.",
                        "Provide a short summary followed by three key takeaways.",
                    ]
                ),
            ),
            ("human", "Please summarize the following content:\n\n{text}"),
        ]
    )


def build_classification_prompt(labels: list[str]) -> ChatPromptTemplate:
    joined_labels = ", ".join(labels)
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "\n".join(
                    [
                        _base_assistant_rules(),
                        "You are a text classifier.",
                        "Choose exactly one label from the provided list.",
                        "Do not invent new labels.",
                        "Return a valid JSON object that matches the required schema.",
                    ]
                ),
            ),
            (
                "human",
                "Available labels: {labels}\n\nText:\n{text}\n\n"
                "Return the result as JSON with label, reason, "
                "and a confidence score between 0 and 1.",
            ),
        ]
    ).partial(labels=joined_labels)


def build_extraction_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "\n".join(
                    [
                        _base_assistant_rules(),
                        "Extract only entities clearly supported by the input text.",
                        "Keep the summary to one sentence.",
                        "Return a valid JSON object that matches the required schema.",
                        "Return entities as a JSON array of strings.",
                        "Do not return entity objects, key-value pairs, or nested structures.",
                    ]
                ),
            ),
            (
                "human",
                "Extract key entities from the following text and summarize it briefly. "
                "Return JSON only.\n\n{text}",
            ),
        ]
    )
