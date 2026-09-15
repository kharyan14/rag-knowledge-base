from src.rag_kb.evaluate import token_f1

def test_exact_answer(): assert token_f1("30 calendar days", "30 calendar days") == 1.0
def test_partial_answer(): assert 0 < token_f1("30 days", "30 calendar days") < 1
