import argparse, json
from dotenv import load_dotenv
from src.rag_kb.index import build_index
from src.rag_kb.qa import ask
from src.rag_kb.evaluate import run_eval

load_dotenv()
p = argparse.ArgumentParser(description="Traceable FAISS document Q&A")
sub = p.add_subparsers(dest="command", required=True)
i = sub.add_parser("ingest"); i.add_argument("path"); i.add_argument("--index", default="index")
q = sub.add_parser("ask"); q.add_argument("question"); q.add_argument("--index", default="index"); q.add_argument("-k", type=int, default=4)
e = sub.add_parser("eval"); e.add_argument("--questions", default="evals/questions.json"); e.add_argument("--index", default="index")
a = p.parse_args()
if a.command == "ingest": print(json.dumps({"chunks_indexed": build_index(a.path, a.index)}))
elif a.command == "ask":
    r = ask(a.index, a.question, a.k); print(r.text); print("\nSources:"); [print(json.dumps(c, ensure_ascii=False)) for c in r.citations]
else: print(json.dumps(run_eval(a.index, a.questions), indent=2))
