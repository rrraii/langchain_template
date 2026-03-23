# langchain12-templates

[English README](./README.md)

面向工程开发的 LangChain 模板库，提供聊天、Agent、结构化输出、RAG、可配置中间件、事件钩子，以及基于 SQLite 的持久化记忆能力。

这个仓库适合直接克隆后扩展到真实项目中，核心能力包括：

- 面向 OpenAI-compatible API 和 Ollama 的统一 provider 配置
- 高层 `TemplateApp` 门面，减少业务侧样板代码
- Chat、Streaming、Summary、Classification、Extraction、Router、Agent、Memory Agent、RAG 模板
- 结构化 RAG、引用校验与无答案兜底
- LangChain 1.2 Agent middleware：工具调用限制、PII 保护、记忆摘要、模型回退、动态模型选择
- `safe`、`balanced`、`aggressive`、`custom` 中间件 profile
- `memory_agent` 默认使用 SQLite 持久化记忆，并支持按线程清理、复制、裁剪历史
- 稳定的结果 schema，统一携带 `trace_id`、`latency_ms`、fallback 元数据
- hooks 事件流，便于观测、日志集成和二次开发
- CLI、示例、测试、文档、CI 和标准仓库元数据

## 安装

```bash
pip install -r requirements.txt
```

可编辑开发安装：

```bash
pip install -e ".[dev]"
```

中国大陆网络环境可使用清华源：

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

`api_key` 支持两种环境变量写法：

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
print(app.summarize("这里放一段较长文本。"))
print(app.route("请总结这段会议纪要。"))
print(app.run("请总结这段会议纪要。").text)
print(app.agent("帮我计算 (15 + 27) * 3，并告诉我现在时间。" ).final_text)
print(app.doctor_display())
```

### 3. 构建知识库并提问

```python
from lc_templates import create_app

app = create_app()
app.index_file("examples/data/medical_demo.txt")
print(app.ask_rag_rendered("高血压患者应该注意什么？"))
```

### 4. 使用持久化记忆

```python
from lc_templates import create_app

app = create_app()
print(app.memory_agent_text("demo-thread", "我叫小王，我在学 LangChain。"))
print(app.memory_agent_text("demo-thread", "你还记得我叫什么吗？"))
```

## CLI 用法

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

常用 `runtime` 字段：

- `runtime.active_provider`
- `runtime.default_output_mode`
- `runtime.log_level`
- `runtime.third_party_log_level`
- `runtime.log_file`
- `runtime.default_collection_name`
- `runtime.default_persist_directory`
- `runtime.rag_no_answer_message`
- `runtime.memory.*`
- `runtime.middleware.*`

常用 `providers` 字段：

- `providers.<name>.api_key`
- `providers.<name>.base_url`
- `providers.<name>.chat_model`
- `providers.<name>.reasoning_model`
- `providers.<name>.embedding_model`
- `providers.<name>.embedding_dimensions`
- `providers.<name>.request_timeout`
- `providers.<name>.max_retries`

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
- `TemplateApp.clear_memory_thread`
- `TemplateApp.copy_memory_thread`
- `TemplateApp.prune_memory_threads`
- `TemplateApp.ask_rag_structured`
- `TemplateApp.ask_rag_rendered`
- `TemplateApp.index_file`
- `TemplateApp.doctor`
- `TemplateApp.doctor_recommendations`
- `TemplateApp.doctor_recommended_profile`
- `TemplateApp.init_config`
- `TemplateApp.on_event`

## 结果 Schema

常用 schema：

- `ResultEnvelope`
- `HookEvent`
- `WorkflowExecutionResult`
- `AgentExecutionResult`
- `ClassificationResult`
- `ExtractionResult`
- `GroundedAnswer`
- `KnowledgeBaseBuildResult`
- `MemoryThreadOperationResult`
- `RouteDecision`
- `TaskBundleResult`

基于 `ResultEnvelope` 的结果对象统一包含：

- `trace_id`
- `latency_ms`
- `status`
- `fallback_used`
- `error_reason`
- `meta`

## 日志与诊断

推荐配置：

```yaml
runtime:
  log_level: INFO
  third_party_log_level: WARNING
  log_file: logs/lc_templates.log
```

说明：

- 框架日志可以同时输出到控制台和日志文件
- 第三方 SDK 的噪音日志由 `third_party_log_level` 控制
- `doctor()` 会返回 `warnings` 和 `recommendations`
- `doctor()` 还会给出 `recommended_middleware_profile`
- `doctor_display(verbose=True)` 是本地排查问题最快的入口
- hooks 提供了代码级观测能力

## 示例

`examples/` 目录包含：

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

- 降低首次接入门槛
- 通过 prompt 约束、schema 校验和 fallback 提升弱模型稳定性
- 为业务侧提供简洁、统一的高层 API
- 在保留 raw 调试信息的同时返回稳定、可消费的结果对象
- 保持配置、示例、CLI、文档、测试长期一致
