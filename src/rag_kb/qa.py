from dataclasses import dataclass
import os
from langchain_openai import ChatOpenAI
from .index import load_index

@dataclass
class Answer:
    text: str
    citations: list[dict]

SYSTEM = """Answer only from the supplied passages. If the answer is absent, say 'I don't know based on the indexed documents.' Cite claims with [S1], [S2], etc. Do not invent citations."""

def ask(index_path: str, question: str, k: int = 4) -> Answer:
    docs = load_index(index_path).similarity_search(question, k=k)
    passages, citations = [], []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"S{i}"
        passages.append(f"[{label}] source={source}; page={page + 1 if isinstance(page, int) else 'n/a'}\n{doc.page_content}")
        citations.append({"id": label, "source": source, "page": page + 1 if isinstance(page, int) else None, "chunk_id": doc.metadata.get("chunk_id"), "passage": doc.page_content})
    prompt = f"Question: {question}\n\nPassages:\n" + "\n\n".join(passages)
    response = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0).invoke([( "system", SYSTEM), ("human", prompt)])
    return Answer(text=response.content, citations=citations)
