from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED = {".pdf", ".txt", ".md"}

def load_documents(path: str) -> list[Document]:
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    docs: list[Document] = []
    for file in files:
        if file.suffix.lower() not in SUPPORTED:
            continue
        loaded = PyPDFLoader(str(file)).load() if file.suffix.lower() == ".pdf" else TextLoader(str(file), encoding="utf-8").load()
        for doc in loaded:
            doc.metadata["source"] = str(file)
        docs.extend(loaded)
    if not docs:
        raise ValueError(f"No supported documents found in {path}")
    return docs
