
import unittest

from lc_templates.rag.hybrid import merge_with_rrf


class SmokeTests(unittest.TestCase):
    def test_merge_with_rrf_returns_docs(self):
        docs = merge_with_rrf(
            query="test",
            dense_search=lambda _query, _k: [],
            sparse_search=lambda _query, _k: [],
            k=3,
        )
        self.assertEqual(docs, [])
