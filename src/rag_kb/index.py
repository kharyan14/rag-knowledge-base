from pathlib import Path
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .loaders import load_documents

def build_index(input_path: str, output_path: str, chunk_size: int = 800, chunk_overlap: int = 120) -> int:
    docs = load_documents(input_path)
    chunks = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    store = FAISS.from_documents(chunks, OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")))
    Path(output_path).mkdir(parents=True, exist_ok=True)
    store.save_local(output_path)
    return len(chunks)

def load_index(path: str) -> FAISS:
    return FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
