from __future__ import annotations

from collections.abc import Callable

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def create_vector_retriever(vector_store, k: int = 4):
    """Create a dense retriever from the configured vector store."""
    return vector_store.as_retriever(search_kwargs={"k": k})


class ChineseBM25Retriever:
    """Simple BM25 retriever for Chinese text using jieba tokenization."""

    def __init__(
        self,
        documents: list[Document],
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.documents = documents
        self.tokenizer = tokenizer or jieba.lcut
        self.corpus = [self.tokenizer(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def get_relevant_documents(self, query: str, k: int = 4) -> list[Document]:
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

        results: list[Document] = []
        for index, score in ranked[:k]:
            doc = self.documents[index]
            enriched = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": float(score)},
            )
            results.append(enriched)
        return results
