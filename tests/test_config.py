
import os
import tempfile
import textwrap
import unittest

from lc_templates.app import TemplateApp
from lc_templates.core.config import get_settings


class ConfigTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("LC_TEMPLATES_CONFIG", None)
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        get_settings.cache_clear()

    def test_template_app_sets_global_config_path_for_subsequent_calls(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as file:
            file.write(
                textwrap.dedent(
                    """
                    runtime:
                      active_provider: ollama
                      rerank_provider: ollama
                      http_proxy: http://127.0.0.1:7890
                      https_proxy: http://127.0.0.1:7890
                      default_collection_name: custom_collection
                      default_persist_directory: custom/index
                      top_k: 7
                      chunk_size: 500
                      chunk_overlap: 100
                      hybrid_rrf_k: 60
                      response_language: zh-CN
                      response_format: markdown
                      answer_style: balanced
                      default_output_mode: json
                      log_level: DEBUG
                      third_party_log_level: ERROR
                      log_file: logs/test.log
                      max_citations: 3
                      routing_confidence_threshold: 0.55
                      rag_no_answer_message: "N/A"

                    providers:
                      ollama:
                        type: ollama
                        enabled: true
                        api_key: ""
                        base_url: http://localhost:11434
                        chat_model: qwen3:4b
                        reasoning_model: qwen3:4b
                        embedding_model: bge-m3
                        embedding_dimensions: 1024
                        rerank_model: ""
                        temperature: 0.1
                        request_timeout: 60
                        max_retries: 2
                    """
                )
            )
            temp_path = file.name

        try:
            app = TemplateApp(config_path=temp_path)
            self.assertEqual(app.settings.runtime.top_k, 7)
            self.assertEqual(app.settings.runtime.default_output_mode, "json")
            self.assertEqual(app.settings.runtime.log_level, "DEBUG")
            self.assertEqual(app.settings.runtime.third_party_log_level, "ERROR")
            self.assertEqual(app.settings.runtime.log_file, "logs/test.log")
            self.assertEqual(app.settings.runtime.http_proxy, "http://127.0.0.1:7890")
            self.assertEqual(app.settings.runtime.https_proxy, "http://127.0.0.1:7890")
            self.assertEqual(app.settings.providers.ollama.embedding_dimensions, 1024)
            self.assertEqual(os.environ.get("LC_TEMPLATES_CONFIG"), temp_path)
            self.assertEqual(os.environ.get("HTTP_PROXY"), "http://127.0.0.1:7890")
            self.assertEqual(os.environ.get("HTTPS_PROXY"), "http://127.0.0.1:7890")
            self.assertEqual(get_settings().runtime.default_collection_name, "custom_collection")
        finally:
            os.unlink(temp_path)
