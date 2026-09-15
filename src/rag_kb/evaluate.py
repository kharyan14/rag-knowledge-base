import json, re
from pathlib import Path
from .index import load_index
from .qa import ask

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def token_f1(prediction: str, expected: str) -> float:
    p, e = _tokens(prediction), _tokens(expected)
    if not p or not e: return float(p == e)
    overlap = len(p & e)
    precision, recall = overlap / len(p), overlap / len(e)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

def run_eval(index_path: str, questions_path: str, k: int = 4) -> dict:
    cases = json.loads(Path(questions_path).read_text())
    store, rows = load_index(index_path), []
    for case in cases:
        retrieved = store.similarity_search(case["question"], k=k)
        retrieved_sources = {Path(d.metadata.get("source", "")).name for d in retrieved}
        expected_sources = set(case["expected_sources"])
        relevance = len(retrieved_sources & expected_sources) / max(1, len(expected_sources))
        result = ask(index_path, case["question"], k=k)
        rows.append({"question": case["question"], "retrieval_relevance": relevance, "answer_accuracy_token_f1": token_f1(result.text, case["expected_answer"]), "answer": result.text})
    return {"cases": rows, "mean_retrieval_relevance": sum(r["retrieval_relevance"] for r in rows)/len(rows), "mean_answer_accuracy_token_f1": sum(r["answer_accuracy_token_f1"] for r in rows)/len(rows)}
