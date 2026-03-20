# langchain12-templates

`langchain12-templates` 是一个偏工程化的 LangChain 模板仓，目标是让项目在“能跑通”之外，也具备更清晰的结构、更稳的默认行为和更好的扩展点。

## 包含能力

- 统一的 provider 模型工厂
- 高层 `TemplateApp` 门面
- 聊天、Agent、结构化输出、RAG 模板
- CLI 与自检命令
- 配置优先的运行时行为
- 测试与仓库自动化配套

## 推荐入口

```python
from lc_templates import create_app

app = create_app()
print(app.chat("请说明这个模板仓适合做什么。"))
```

安装与配置请先阅读根目录 [README.zh-CN](../README.zh-CN.md)。
