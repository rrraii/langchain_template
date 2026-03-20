import unittest
from unittest.mock import patch

from lc_templates.core.config import ProviderSettings
from lc_templates.core.models import build_embeddings


class ModelFactoryTests(unittest.TestCase):
    def test_openai_compatible_non_openai_embeddings_disable_tiktoken(self):
        provider = ProviderSettings(
            type="openai_compatible",
            api_key="sk-test",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            embedding_model="text-embedding-v1",
        )
        with patch("lc_templates.core.models._resolve_provider", return_value=("qwen", provider)):
            with patch("lc_templates.core.models.OpenAIEmbeddings") as mocked:
                build_embeddings(provider_name="qwen")

        mocked.assert_called_once()
        kwargs = mocked.call_args.kwargs
        self.assertFalse(kwargs["tiktoken_enabled"])
        self.assertFalse(kwargs["check_embedding_ctx_length"])

    def test_openai_embeddings_keep_default_token_handling(self):
        provider = ProviderSettings(
            type="openai_compatible",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            embedding_model="text-embedding-3-small",
        )
        with patch("lc_templates.core.models._resolve_provider", return_value=("openai", provider)):
            with patch("lc_templates.core.models.OpenAIEmbeddings") as mocked:
                build_embeddings(provider_name="openai")

        mocked.assert_called_once()
        kwargs = mocked.call_args.kwargs
        self.assertNotIn("tiktoken_enabled", kwargs)
        self.assertNotIn("check_embedding_ctx_length", kwargs)
