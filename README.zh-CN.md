# langchain12-templates

[English README](./README.md)

面向工程开发的 LangChain 模板库，覆盖聊天、Agent、结构化输出、RAG 工作流，以及可配置的 LangChain 1.2 middleware 能力。

这个仓库的目标是让你可以更快地 clone、配置、扩展，并直接接入真实项目。它提供：

- 基于 provider 的模型工厂，支持 OpenAI Compatible API 和 Ollama
- 高层 `TemplateApp` 门面，减少业务方直接拼底层链路的工作量
- 聊天、流式输出、摘要、分类、抽取、路由、Agent、记忆 Agent、RAG 模板
- 带引用校验和无答案兜底的结构化 RAG
- 可配置的 LangChain 1.2 agent middleware
  - 工具调用限制
  - PII 保护
  - memory summarization
  - model fallback
  - dynamic model selection
- middleware profile：`safe`、`balanced`、`aggressive`、`custom`
- 稳定的结果 schema，统一包含 `trace_id`、`latency_ms`、fallback 元信息和结构化负载
- 事件 hooks，可用于埋点、监控、调试和自定义集成
- CLI、示例、测试、文档、CI 和开源仓库配套文件

## 安装

```bash
pip install -r requirements.txt
```

可编辑安装：

```bash
pip install -e ".[dev]"
```

中国大陆网络环境可使用：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 快速开始

### 1. 初始化配置

```bash
lc-templates init-config
```

或者手动复制：

- `config/config.example.yaml` -> `config/config.yaml`

`api_key` 支持两种写法：

```yaml
api_key: DASHSCOPE_API_KEY
```

```yaml
api_key: ${DASHSCOPE_API_KEY}
```

### 2. 使用高层门面

```python
from lc_templates import create_app

app = create_app()

print(app.chat("这个模板库适合做什么？"))
print(app.summarize("这里放一段长文本。"))
print(app.route("请帮我总结这段会议纪要。"))
print(app.run("请帮我总结这段会议纪要。").text)
print(app.agent("帮我计算 (15 + 27) * 3，并告诉我现在时间。").final_text)
print(app.doctor_display())
```

### 3. 构建并查询知识库

```python
from lc_templates import create_app

app = create_app()
app.index_file("examples/data/medical_demo.txt")
print(app.ask_rag_rendered("高血压患者平时需要注意什么？"))
```

### 4. 订阅事件 hooks

```python
from lc_templates import create_app

app = create_app()
app.on_event(lambda event: print(event.name, event.trace_id, event.payload))
print(app.classify_label("请总结这段会议纪要。", ["summarize", "chat"]))
```

## CLI 使用示例

```bash
lc-templates init-config
lc-templates chat "Explain what RAG is."
lc-templates summarize "Summarize this text."
lc-templates classify "Summarize this memo." --labels rag extract summarize chat
lc-templates route "Please summarize this meeting note."
lc-templates run "Please summarize this meeting note."
lc-templates run "What should patients with hypertension pay attention to?" --use-rag --collection-name demo_collection__qwen__text-embedding-v1__1536d
lc-templates agent "Calculate (15 + 27) * 3 and tell me the current time." --output json
lc-templates doctor --output verbose
lc-templates config --output json
lc-templates index examples/data/medical_demo.txt
lc-templates rag "What should patients with hypertension pay attention to?"
```

## 常用配置

`config/config.yaml` 里最常用的字段包括：

- `runtime.active_provider`
- `runtime.default_output_mode`
- `runtime.log_level`
- `runtime.third_party_log_level`
- `runtime.log_file`
- `runtime.default_collection_name`
- `runtime.default_persist_directory`
- `runtime.routing_confidence_threshold`
- `runtime.rag_no_answer_message`
- `runtime.middleware`
- `providers.<name>.api_key`
- `providers.<name>.base_url`
- `providers.<name>.chat_model`
- `providers.<name>.reasoning_model`
- `providers.<name>.embedding_model`
- `providers.<name>.embedding_dimensions`
- `providers.<name>.request_timeout`
- `providers.<name>.max_retries`

middleware 常用字段：

- `runtime.middleware.profile`
- `runtime.middleware.tool_call_limit_enabled`
- `runtime.middleware.tool_call_limit`
- `runtime.middleware.model_fallback_enabled`
- `runtime.middleware.dynamic_model_selection_enabled`
- `runtime.middleware.dynamic_model_selection_message_threshold`
- `runtime.middleware.pii.*`
- `runtime.middleware.summarization.*`

完整参数说明见：

- `docs/configuration.md`

## 推荐 API

- `lc_templates.create_app`
- `lc_templates.TemplateApp`
- `lc_templates.register_event_hook`
- `lc_templates.unregister_event_hook`
- `lc_templates.clear_event_hooks`
- `TemplateApp.run`
- `TemplateApp.run_display`
- `TemplateApp.agent`
- `TemplateApp.memory_agent`
- `TemplateApp.ask_rag_structured`
- `TemplateApp.ask_rag_rendered`
- `TemplateApp.index_file`
- `TemplateApp.doctor`
- `TemplateApp.doctor_recommendations`
- `TemplateApp.doctor_recommended_profile`
- `TemplateApp.init_config`
- `TemplateApp.on_event`

## 核心结果 Schema

常用 schema：

- `ResultEnvelope`
- `HookEvent`
- `WorkflowExecutionResult`
- `AgentExecutionResult`
- `ClassificationResult`
- `ExtractionResult`
- `GroundedAnswer`
- `KnowledgeBaseBuildResult`
- `RouteDecision`
- `TaskBundleResult`

基于 `ResultEnvelope` 的结果统一包含：

- `trace_id`
- `latency_ms`
- `status`
- `fallback_used`
- `error_reason`
- `meta`

## 日志与诊断

推荐默认配置：

```yaml
runtime:
  log_level: INFO
  third_party_log_level: WARNING
  log_file: logs/lc_templates.log
```

说明：

- 框架日志会输出到控制台，也可以同时写入文件
- 第三方 SDK 噪音日志会通过 `third_party_log_level` 降下来
- `doctor()` 会返回 `warnings` 和 `recommendations`
- `doctor()` 还会返回 `recommended_middleware_profile`
- `doctor_display(verbose=True)` 是最快的本地排查视图
- 事件 hooks 提供了日志之外的代码级可观测性入口

## 示例脚本

`examples/` 中包含：

- `run_app.py`
- `run_basic_chat.py`
- `run_agent.py`
- `run_memory_agent.py`
- `run_structured_output.py`
- `run_rag.py`
- `run_router.py`
- `run_streaming.py`
- `run_tasks.py`
- `run_hybrid_search.py`
- `run_quickstart.py`
- `run_quickstart_rag.py`

## 测试

```bash
pytest -q
```

覆盖率：

```bash
pytest --cov=lc_templates --cov-report=term-missing
```

## 设计目标

- 降低第一次运行和接入成本
- 通过 prompt 约束、schema 校验和 fallback 提升弱模型稳定性
- 给业务团队暴露更干净的顶层 API
- 既保留原始结果用于调试，也返回规范化结果用于工程接入
- 保持配置、示例、CLI、文档、测试长期一致
