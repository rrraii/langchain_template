# API

Recommended public API:

## Factory

- `lc_templates.create_app`
- `lc_templates.register_event_hook`
- `lc_templates.unregister_event_hook`
- `lc_templates.clear_event_hooks`

## Main facade

- `TemplateApp.chat`
- `TemplateApp.classify_label`
- `TemplateApp.classify_text_result`
- `TemplateApp.extract_text`
- `TemplateApp.extract_display`
- `TemplateApp.route_name`
- `TemplateApp.route_display`
- `TemplateApp.summarize`
- `TemplateApp.classify`
- `TemplateApp.extract`
- `TemplateApp.route`
- `TemplateApp.route_result`
- `TemplateApp.run`
  Supports `use_rag=True` plus vector-store options.
- `TemplateApp.run_display`
- `TemplateApp.init_config`
- `TemplateApp.on_event`
- `TemplateApp.clear_event_hooks`
- `TemplateApp.agent`
- `TemplateApp.agent_text`
- `TemplateApp.agent_display`
- `TemplateApp.memory_agent`
- `TemplateApp.memory_agent_text`
- `TemplateApp.memory_agent_display`
- `TemplateApp.clear_memory_thread`
- `TemplateApp.clear_memory_thread_display`
- `TemplateApp.copy_memory_thread`
- `TemplateApp.copy_memory_thread_display`
- `TemplateApp.prune_memory_threads`
- `TemplateApp.prune_memory_threads_display`
- `TemplateApp.run_text_tasks_display`
- `TemplateApp.run_text_tasks_json`
- `TemplateApp.index_file`
- `TemplateApp.ask_rag_rendered`
- `TemplateApp.ask_rag_structured`
- `TemplateApp.ask_rag_from_file`
- `TemplateApp.doctor`
- `TemplateApp.doctor_summary`
- `TemplateApp.doctor_recommendations`
- `TemplateApp.doctor_recommended_profile`

## Common schemas

- `ResultEnvelope`
- Includes `trace_id`, `latency_ms`, fallback markers, and execution metadata.
- `HookEvent`
- `CitationItem`
- `ToolCallRecord`
- `RouteDecision`
- `TaskBundleResult`
- `WorkflowExecutionResult`
- `ClassificationResult`
- `ExtractionResult`
- `AgentExecutionResult`
- `GroundedAnswer`
- `KnowledgeBaseBuildResult`
- `MemoryThreadOperationResult`

## Event hooks

You can subscribe to framework events with either:

- `TemplateApp.on_event(callback)`
- `lc_templates.register_event_hook(callback)`

Callbacks receive a `HookEvent` with:

- `name`
- `level`
- `message`
- `trace_id`
- `meta`
- `payload`
