import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from lc_templates.core.middleware import (
    _looks_like_topic_shift,
    _select_messages_for_current_turn,
    build_agent_middleware,
)


class MiddlewareFactoryTests(unittest.TestCase):
    def test_topic_shift_detection_identifies_unrelated_turn(self):
        previous_messages = [
            HumanMessage(content="Please summarize this meeting note."),
            HumanMessage(content="What were the action items from the meeting?"),
        ]
        self.assertTrue(
            _looks_like_topic_shift(
                "How should I manage my blood pressure?",
                previous_messages,
                0.18,
            )
        )

    def test_select_messages_for_current_turn_keeps_recent_slice(self):
        messages = [HumanMessage(content=f"message-{index}") for index in range(6)]
        selected = _select_messages_for_current_turn(messages, 3)
        self.assertEqual(
            [message.content for message in selected],
            ["message-3", "message-4", "message-5"],
        )

    def test_build_agent_middleware_returns_tool_limit_by_default(self):
        middleware = build_agent_middleware(has_memory=False)
        self.assertTrue(
            any(item.__class__.__name__ == "ToolCallLimitMiddleware" for item in middleware)
        )

    def test_build_agent_middleware_adds_memory_summarization(self):
        fake_model = type("FakeModel", (), {"_llm_type": "openai-chat"})()
        with patch("lc_templates.core.middleware.build_chat_model", return_value=fake_model):
            middleware = build_agent_middleware(has_memory=True)
        self.assertTrue(
            any(item.__class__.__name__ == "SummarizationMiddleware" for item in middleware)
        )

    def test_build_agent_middleware_can_be_disabled(self):
        settings = type(
            "Settings",
            (),
            {
                "runtime": type(
                    "Runtime",
                    (),
                    {
                        "middleware": type(
                            "Middleware",
                            (),
                            {
                                "enabled": False,
                            },
                        )()
                    },
                )()
            },
        )()
        with patch("lc_templates.core.middleware.get_settings", return_value=settings):
            self.assertEqual(build_agent_middleware(), [])

    def test_build_agent_middleware_adds_pii_and_fallback(self):
        middleware_settings = type(
            "Middleware",
            (),
            {
                "enabled": True,
                "tool_call_limit_enabled": False,
                "tool_call_limit": 6,
                "model_fallback_enabled": True,
                "dynamic_model_selection_enabled": False,
                "dynamic_model_selection_message_threshold": 800,
                "pii": type(
                    "PII",
                    (),
                    {
                        "enabled": True,
                        "pii_types": ["email"],
                        "strategy": "redact",
                        "apply_to_input": True,
                        "apply_to_output": False,
                        "apply_to_tool_results": False,
                    },
                )(),
                "summarization": type(
                    "Summary",
                    (),
                    {"enabled": False, "trigger_messages": 24, "keep_messages": 12},
                )(),
                "context_guard": type(
                    "ContextGuard",
                    (),
                    {
                        "enabled": True,
                        "similarity_threshold": 0.18,
                        "recent_messages": 4,
                        "preserve_profile_facts": True,
                    },
                )(),
            },
        )()
        provider = type("Provider", (), {"reasoning_model": "r1", "chat_model": "chat"})()
        settings = type(
            "Settings",
            (),
            {
                "runtime": type("Runtime", (), {"middleware": middleware_settings})(),
                "get_active_provider": lambda self: provider,
            },
        )()
        with patch("lc_templates.core.middleware.get_settings", return_value=settings):
            with patch("lc_templates.core.middleware.build_chat_model", return_value=object()):
                with patch(
                    "lc_templates.core.middleware.build_reasoning_model",
                    return_value=object(),
                ):
                    middleware = build_agent_middleware()
        names = [item.__class__.__name__ for item in middleware]
        self.assertIn("PIIMiddleware", names)
        self.assertIn("ModelFallbackMiddleware", names)
        self.assertNotIn("CurrentTurnPriorityMiddleware", names)

    def test_build_agent_middleware_adds_context_guard_for_memory_agents(self):
        fake_model = type("FakeModel", (), {"_llm_type": "openai-chat"})()
        with patch("lc_templates.core.middleware.build_chat_model", return_value=fake_model):
            middleware = build_agent_middleware(has_memory=True)
        self.assertTrue(
            any(item.__class__.__name__ == "CurrentTurnPriorityMiddleware" for item in middleware)
        )

    def test_dynamic_model_selection_switches_to_reasoning_model(self):
        middleware_settings = type(
            "Middleware",
            (),
            {
                "enabled": True,
                "tool_call_limit_enabled": False,
                "tool_call_limit": 6,
                "model_fallback_enabled": False,
                "dynamic_model_selection_enabled": True,
                "dynamic_model_selection_message_threshold": 10,
                "pii": type(
                    "PII",
                    (),
                    {
                        "enabled": False,
                        "pii_types": [],
                        "strategy": "redact",
                        "apply_to_input": True,
                        "apply_to_output": False,
                        "apply_to_tool_results": False,
                    },
                )(),
                "summarization": type(
                    "Summary",
                    (),
                    {"enabled": False, "trigger_messages": 24, "keep_messages": 12},
                )(),
                "context_guard": type(
                    "ContextGuard",
                    (),
                    {
                        "enabled": False,
                        "similarity_threshold": 0.18,
                        "recent_messages": 4,
                        "preserve_profile_facts": True,
                    },
                )(),
            },
        )()
        provider = type("Provider", (), {"reasoning_model": "r1", "chat_model": "chat"})()
        settings = type(
            "Settings",
            (),
            {
                "runtime": type("Runtime", (), {"middleware": middleware_settings})(),
                "get_active_provider": lambda self: provider,
            },
        )()
        captured = {}

        def handler(request):
            captured["model"] = request.model
            return "ok"

        request = SimpleNamespace(
            model="chat-model",
            messages=[HumanMessage(content="x" * 20)],
            override=lambda **kwargs: SimpleNamespace(
                model=kwargs["model"],
                messages=[HumanMessage(content="x" * 20)],
            ),
        )

        with patch("lc_templates.core.middleware.get_settings", return_value=settings):
            with patch(
                "lc_templates.core.middleware.build_reasoning_model",
                return_value="reasoning-model",
            ):
                middleware = build_agent_middleware()
                dynamic = next(
                    item
                    for item in middleware
                    if item.__class__.__name__ == "DynamicModelSelectionMiddleware"
                )
                dynamic.wrap_model_call(request, handler)

        self.assertEqual(captured["model"], "reasoning-model")
