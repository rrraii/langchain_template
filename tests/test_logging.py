import logging
import tempfile
import unittest
from pathlib import Path

from lc_templates.core.config import RuntimeSettings
from lc_templates.core.logging import configure_logging_from_runtime, setup_logging


class LoggingTests(unittest.TestCase):
    def test_configure_logging_from_runtime_applies_third_party_level(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "app.log"
            runtime = RuntimeSettings(
                log_level="INFO",
                third_party_log_level="ERROR",
                log_file=str(log_path),
            )

            configure_logging_from_runtime(runtime)

            self.assertEqual(logging.getLogger("openai").level, logging.ERROR)
            self.assertEqual(logging.getLogger("httpx").level, logging.ERROR)
            setup_logging(force=True)
