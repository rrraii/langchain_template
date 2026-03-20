from __future__ import annotations

from langchain_core.documents import Document

from lc_templates.core.config import get_settings
from lc_templates.core.models import build_openai_compatible_client


class OpenAICompatibleReranker:
    """适合 Qwen / DashScope / DeepSeek / 其他 OpenAI Compatible rerank 接口。"""

    def __init__(self, model: str | None = None, provider_name: str | None = None) -> None:
        settings = get_settings()
        self.provider_name = provider_name or settings.get_rerank_provider_name()
        self.provider = settings.get_provider(self.provider_name)
        self.model = model or self.provider.rerank_model

        if self.provider.type == "openai_compatible":
            self.client = build_openai_compatible_client(
                provider_name=self.provider_name, for_rerank=True
            )
        else:
            self.client = None

    def rerank(self, query: str, documents: list[Document], top_n: int = 4) -> list[Document]:
        if not self.model or not self.client:
            return documents[:top_n]

        response = self.client.post(
            "/rerank",
            cast_to=dict,
            body={
                "model": self.model,
                "query": query,
                "documents": [doc.page_content for doc in documents],
                "top_n": top_n,
            },
        )

        index_to_doc = {index: doc for index, doc in enumerate(documents)}
        ranked = response.get("results", [])
        return [index_to_doc[item["index"]] for item in ranked if item["index"] in index_to_doc]
