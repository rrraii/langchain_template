import _bootstrap as _bootstrap  # noqa: F401
from _args import build_example_parser, resolve_output_mode
from langchain_core.documents import Document

from lc_templates.core.output import to_pretty_json
from lc_templates.rag.hybrid import merge_with_rrf
from lc_templates.rag.retrievers import ChineseBM25Retriever


def _mock_dense_search(_: str, k: int):
    docs = [
        Document(
            page_content="Patients with hypertension should monitor blood pressure regularly.",
            metadata={"source": "dense_1"},
        ),
        Document(
            page_content="Patients with diabetes should monitor fasting blood sugar.",
            metadata={"source": "dense_2"},
        ),
        Document(
            page_content="Fever with cough may indicate a respiratory infection.",
            metadata={"source": "dense_3"},
        ),
    ]
    return docs[:k]


if __name__ == "__main__":
    args = build_example_parser("Run the hybrid retrieval example.").parse_args()
    output_mode = resolve_output_mode(args)
    documents = [
        Document(
            page_content="Hypertension requires follow-up visits and lower salt intake.",
            metadata={"source": "bm25_1"},
        ),
        Document(
            page_content="Fever with cough is common in respiratory infections.",
            metadata={"source": "bm25_2"},
        ),
        Document(
            page_content="Diabetes requires long-term diet and exercise management.",
            metadata={"source": "bm25_3"},
        ),
    ]
    bm25 = ChineseBM25Retriever(documents)
    results = merge_with_rrf(
        query="What should I do about fever and cough?",
        dense_search=_mock_dense_search,
        sparse_search=bm25.get_relevant_documents,
        k=3,
    )

    if output_mode == "json":
        print(
            to_pretty_json(
                [
                    {
                        "rank": index,
                        "source": item.metadata.get("source", f"doc_{index}"),
                        "page_content": item.page_content,
                        "metadata": item.metadata,
                    }
                    for index, item in enumerate(results, start=1)
                ]
            )
        )
    else:
        for index, item in enumerate(results, start=1):
            source = item.metadata.get("source", f"doc_{index}")
            if output_mode == "verbose":
                print(f"[{index}] source={source}")
                print(item.page_content)
                print()
            else:
                print(f"[{index}] {source}: {item.page_content}")
