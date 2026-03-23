import unittest

from lc_templates.core.hooks import (
    clear_event_hooks,
    emit_event,
    register_event_hook,
    unregister_event_hook,
)
from lc_templates.core.schemas import ExecutionMetadata


class HookTests(unittest.TestCase):
    def setUp(self):
        clear_event_hooks()

    def tearDown(self):
        clear_event_hooks()

    def test_emit_event_dispatches_to_registered_hook(self):
        received = []
        register_event_hook(received.append)

        event = emit_event(
            "demo.event",
            message="demo",
            trace_id="trace-1",
            meta=ExecutionMetadata(provider_name="ollama", operation="demo"),
            payload={"ok": True},
        )

        self.assertEqual(event.name, "demo.event")
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].trace_id, "trace-1")
        self.assertEqual(received[0].payload["ok"], True)

    def test_unregister_event_hook_stops_future_dispatch(self):
        received = []

        def callback(event):
            received.append(event)

        register_event_hook(callback)
        unregister_event_hook(callback)
        emit_event("demo.event")

        self.assertEqual(received, [])
