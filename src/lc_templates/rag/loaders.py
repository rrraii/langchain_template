from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document


def load_documents(path: str) -> list[Document]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return TextLoader(str(file_path), encoding="utf-8").load()
    if suffix == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    if suffix in {".docx", ".doc"}:
        return UnstructuredWordDocumentLoader(str(file_path)).load()

    raise ValueError(f"暂不支持的文件类型: {suffix}")
