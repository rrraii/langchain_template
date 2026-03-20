# langchain12-templates

[English README](./README.md)

这是一个面向工程开发的 LangChain 模板仓，覆盖聊天、Agent、结构化输出、RAG 与高层应用封装。

它的目标不是只演示“怎么跑通”，而是尽量提供一套适合业务项目直接复用的基础骨架。仓库提供：

- 面向 OpenAI Compatible 与 Ollama 的统一模型工厂
- 高层 `TemplateApp` 门面，降低业务接入复杂度
- 聊天、流式、摘要、分类、抽取、路由、Agent 等常见模板
- 带引用校验与无答案回退的结构化 RAG
- 可配置的响应语言、格式、风格、重试、超时、分块、路由阈值
- 支持 `pytest` 与标准库 `unittest` 的测试体系

## 为什么它不只是一个演示仓库

- 提供高层 facade 和 CLI，业务代码可以直接接入
- 自带健康检查，能更早发现配置和 provider 可用性问题
- 结构化输出有 fallback 解析，对弱模型更友好
- RAG 带引用过滤和无答案回退，尽量减少“看起来像对”的幻觉
- 打包、文档、测试、仓库配套更完整，适合长期维护

## 核心特性

- 标准包导入：统一使用 `lc_templates`
- 配置优先：运行时行为统一从 `config/config.yaml` 读取
- Agent / RAG 都有更稳定的工程化返回对象
- 提供 CLI、自检命令、示例脚本与测试
- 仓库结构完整，适合作为 GitHub 模板仓或业务项目起点

## 安装

创建虚拟环境后安装依赖：

```bash
pip install -r requirements.txt
```

如果你更偏好可编辑安装和开发依赖：

```bash
pip install -e ".[dev]"
```

如果你在国内，也可以使用镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

注意镜像源要写成 `-i` 参数，不要直接把网址放在命令最后。

## 快速开始

### 1. 配置 Provider

复制 `config/config.example.yaml` 为 `config/config.yaml`，然后填写对应 provider。

支持这两种环境变量写法：

```yaml
api_key: DASHSCOPE_API_KEY
```

```yaml
api_key: ${DASHSCOPE_API_KEY}
```

### 2. 使用高层应用入口

```python
from lc_templates import create_app

app = create_app()

print(app.chat("请说明这个模板仓适合做什么。"))
print(app.summarize("这里是一段长文本。"))
print(app.route("请总结这段会议纪要。"))
print(app.agent("帮我计算 (15 + 27) * 3，并告诉我现在时间。").final_text)
print(app.version())
print(app.config()["runtime"]["active_provider"])
print(app.doctor().model_dump())
```

### 3. 构建并查询知识库

```python
from lc_templates import create_app

app = create_app()
app.index_file("examples/data/medical_demo.txt")

answer = app.ask_rag_rendered("高血压患者平时需要注意什么？")
print(answer)

one_shot = app.ask_rag_from_file(
    "examples/data/medical_demo.txt",
    "高血压患者平时需要注意什么？",
)
print(one_shot)
```

## CLI 用法

安装后可以直接使用 CLI：

```bash
lc-templates chat "请解释什么是 RAG。"
lc-templates summarize "请总结这段文本。"
lc-templates classify "请总结这段会议纪要。" --labels rag extract summarize chat
lc-templates route "请总结这段会议纪要。"
lc-templates agent "帮我计算 (15 + 27) * 3，并告诉我现在时间。" --output json
lc-templates classify "请总结这段会议纪要。" --labels rag extract summarize chat --output verbose
lc-templates version
lc-templates config
lc-templates doctor
lc-templates index examples/data/medical_demo.txt
lc-templates rag "高血压患者平时需要注意什么？"
lc-templates rag "高血压患者平时需要注意什么？" --output json
```

也可以通过模块方式运行：

```bash
python -m lc_templates chat "你好"
```

## 配置说明

主配置文件位于 `config/config.yaml`。

常用运行时字段：

- `active_provider`
- `rerank_provider`
- `http_proxy`
- `https_proxy`
- `default_collection_name`
- `default_persist_directory`
- `top_k`
- `chunk_size`
- `chunk_overlap`
- `hybrid_rrf_k`
- `response_language`
- `response_format`
- `answer_style`
- `default_output_mode`
- `log_level`
- `third_party_log_level`
- `log_file`
- `max_citations`
- `routing_confidence_threshold`
- `rag_no_answer_message`

常用 provider 字段：

- `base_url`
- `api_key`
- `chat_model`
- `reasoning_model`
- `embedding_model`
- `embedding_dimensions`
- `rerank_model`
- `temperature`
- `request_timeout`
- `max_retries`

示例脚本的输出模式有两种控制方式：

- 在 `config/config.yaml` 里设置 `runtime.default_output_mode`
- 运行时用 `--output concise`、`--output verbose` 或 `--output json` 临时覆盖

CLI 对应命令也遵循同样的输出模式约定。

## 项目结构

```text
langchain12-templates/
├─ .github/
├─ config/
├─ examples/
├─ src/lc_templates/
│  ├─ agents/
│  ├─ chains/
│  ├─ core/
│  ├─ rag/
│  ├─ tools/
│  ├─ workflows/
│  ├─ app.py
│  ├─ cli.py
│  └─ __main__.py
├─ tests/
├─ CHANGELOG.md
├─ CONTRIBUTING.md
├─ SECURITY.md
├─ LICENSE
├─ Makefile
├─ pyproject.toml
├─ README.md
└─ README.zh-CN.md
```

## 推荐 API

推荐入口：

- `lc_templates.create_app`
- `lc_templates.TemplateApp`

高层常用方法：

- `app.chat`
- `app.version`
- `app.config`
- `app.summarize`
- `app.classify`
- `app.extract`
- `app.route`
- `app.agent`
- `app.memory_agent`
- `app.index_file`
- `app.ask_rag_rendered`
- `app.ask_rag_structured`
- `app.ask_rag_from_file`
- `app.doctor`

底层常用模块：

- `lc_templates.chains.basic_chat`
- `lc_templates.chains.structured_output`
- `lc_templates.agents.basic_agent`
- `lc_templates.agents.memory_agent`
- `lc_templates.rag.pipeline`

## 测试

使用 `pytest`：

```bash
pytest -q
```

查看覆盖率：

```bash
pytest --cov=lc_templates --cov-report=term-missing
```

执行 pre-commit：

```bash
pre-commit run --all-files
```

或者使用标准库：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## 仓库级规范

当前仓库已经包含：

- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- GitHub issue 模板
- GitHub PR 模板
- GitHub Actions CI
- `.editorconfig`
- `.pre-commit-config.yaml`
- `docs/` 下的 MkDocs 文档骨架

## 示例

示例脚本位于 `examples/`：

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

## 设计目标

- 尽量通过提示词约束、Schema 校验和回退逻辑降低模型不稳定性
- 把配置集中化，便于业务项目统一管理
- 提供清晰的高层 API，降低下游团队接入成本
- 保留 raw 结果便于调试，同时返回工程化对象便于生产使用
- 提供足够的测试、示例与仓库级文档，支持长期维护
