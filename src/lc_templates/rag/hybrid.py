from __future__ import annotations

from collections.abc import Callable

from langchain_core.documents import Document


def _doc_key(doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    return f"{source}::{page}::{doc.page_content[:80]}"


# 推荐在工作中优先使用 RRF 融合，而不是手写分值归一化权重。
def merge_with_rrf(
    query: str,
    dense_search: Callable[[str, int], list[Document]],
    sparse_search: Callable[[str, int], list[Document]],
    k: int = 4,
    rrf_k: int = 60,
) -> list[Document]:
    dense_docs = dense_search(query, k)
    sparse_docs = sparse_search(query, k)

    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs, start=1):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_docs, start=1):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:k]]
