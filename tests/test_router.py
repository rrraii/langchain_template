
import unittest
from unittest.mock import patch

from lc_templates.core.config import get_settings
from lc_templates.core.schemas import ClassificationResult
from lc_templates.workflows import router


class RouterTests(unittest.TestCase):
    def setUp(self):
        get_settings.cache_clear()

    def test_route_task_falls_back_to_chat_when_confidence_is_low(self):
        with patch.object(
            router,
            "classify_text",
            return_value=ClassificationResult(label="rag", reason="weak match", confidence=0.2),
        ):
            self.assertEqual(router.route_task("帮我查一下资料"), "route:chat")

    def test_route_task_uses_predicted_route_when_confidence_is_high(self):
        with patch.object(
            router,
            "classify_text",
            return_value=ClassificationResult(label="summarize", reason="clear", confidence=0.9),
        ):
            self.assertEqual(router.route_task("请总结一下这段文字"), "route:summarize")
