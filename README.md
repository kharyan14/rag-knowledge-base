# RAG Knowledge Base

A traceable document Q&A pipeline built with Python, LangChain, OpenAI embeddings and FAISS. It loads PDF, Markdown and text files, chunks them, creates a local semantic index, answers only from retrieved passages, and returns the exact passages used as citations.

## What it demonstrates
- Separate ingestion and retrieval flows
- Recursive chunking with overlap
- OpenAI embeddings stored in a local FAISS index
- Grounded answers with `[S1]`-style citations and full source passages
- A fixed eval set measuring source retrieval relevance and token-level answer accuracy

## Setup
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
```

## Run
```bash
python app.py ingest data/sample_docs
python app.py ask "How long do I have to request a refund?"
python app.py eval
pytest -q
```

The first three commands require `OPENAI_API_KEY`. Tests for the deterministic scoring logic run without a key. The FAISS directory is local and excluded from Git.

## Eval interpretation
`retrieval_relevance` is source recall for each fixed question: did retrieval return the expected source file? `answer_accuracy_token_f1` measures token overlap against a reference answer. It is transparent and repeatable, but not a substitute for human or LLM-judge evaluation on nuanced answers.

## Structure
```
app.py                         CLI
src/rag_kb/loaders.py          file loading
src/rag_kb/index.py            chunking, embeddings, FAISS
src/rag_kb/qa.py               retrieval, grounded generation, citations
src/rag_kb/evaluate.py         eval harness
data/sample_docs/              safe sample knowledge base
evals/questions.json           fixed test set
```

## Security
Keys are read from environment variables. `.env` and generated indexes are gitignored. Never commit a real key.
