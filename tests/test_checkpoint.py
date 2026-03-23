import os
import sqlite3
import tempfile
import unittest

from lc_templates.core.checkpoint import SqliteCheckpointSaver, build_memory_checkpointer
from lc_templates.core.config import get_settings


class CheckpointTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("LC_TEMPLATES_CONFIG", None)
        get_settings.cache_clear()

    def test_sqlite_checkpoint_saver_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            saver = SqliteCheckpointSaver(os.path.join(temp_dir, "memory.sqlite"))
            try:
                config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
                checkpoint = {
                    "v": 1,
                    "id": "0001",
                    "ts": "2026-03-23T00:00:00Z",
                    "channel_values": {"messages": ["hello"]},
                    "channel_versions": {"messages": "0001.1"},
                    "versions_seen": {},
                    "pending_sends": [],
                    "updated_channels": None,
                }
                saver.put(
                    config,
                    checkpoint,
                    {"source": "input", "step": -1},
                    {"messages": "0001.1"},
                )

                loaded = saver.get_tuple(
                    {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
                )

                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(loaded.checkpoint["channel_values"]["messages"], ["hello"])
                self.assertEqual(loaded.metadata["source"], "input")
                conn = sqlite3.connect(os.path.join(temp_dir, "memory.sqlite"))
                try:
                    row = conn.execute(
                        "SELECT checkpoint_preview, metadata_preview FROM checkpoints"
                    ).fetchone()
                    assert row is not None
                    self.assertIn("messages", row[0])
                    self.assertIn('"source": "input"', row[1])
                finally:
                    conn.close()
            finally:
                saver.close()

    def test_build_memory_checkpointer_uses_sqlite_backend_from_config(self):
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as file:
            file.write(
                """
runtime:
  active_provider: ollama
  rerank_provider: ollama
  default_collection_name: demo_collection
  default_persist_directory: data/index/chroma
  top_k: 4
  chunk_size: 500
  chunk_overlap: 100
  hybrid_rrf_k: 60
  response_language: zh-CN
  response_format: markdown
  answer_style: balanced
  default_output_mode: concise
  log_level: INFO
  third_party_log_level: WARNING
  log_file: null
  max_citations: 3
  routing_confidence_threshold: 0.55
  rag_no_answer_message: "I cannot answer confidently from the provided context."
  memory:
    enabled: true
    backend: sqlite
    sqlite_path: data/state/test-memory.sqlite
    checkpoint_ns: ""

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
            temp_path = file.name

        try:
            os.environ["LC_TEMPLATES_CONFIG"] = temp_path
            get_settings.cache_clear()
            saver = build_memory_checkpointer()
            self.assertIsInstance(saver, SqliteCheckpointSaver)
        finally:
            os.unlink(temp_path)
