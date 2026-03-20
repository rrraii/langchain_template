import unittest

from lc_templates.core.config import resolve_runtime_path
from lc_templates.rag.indexing import resolve_collection_name


class IndexingTests(unittest.TestCase):
    def test_resolve_collection_name_includes_provider_model_and_dimension(self):
        name = resolve_collection_name(provider_name="qwen")

        self.assertEqual(name, "demo_collection__qwen__text-embedding-v1__1536d")

    def test_explicit_collection_name_is_preserved(self):
        name = resolve_collection_name("custom_collection", provider_name="qwen")

        self.assertEqual(name, "custom_collection")

    def test_resolve_runtime_path_uses_project_root_for_relative_paths(self):
        path = resolve_runtime_path("logs/lc_templates.log")

        self.assertIsNotNone(path)
        self.assertTrue(str(path).endswith("logs\\lc_templates.log"))
        self.assertNotIn("examples\\logs", str(path))
