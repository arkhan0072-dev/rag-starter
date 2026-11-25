from app.retrieval import Hit
from eval.run_eval import first_relevant_rank, score

Q = {"question": "?", "source": "a.md", "answer": "25 Days"}


def hit(source: str, content: str) -> Hit:
    return Hit(chunk_id=0, source=source, heading="", content=content, score=0.0)


def test_rank_needs_right_source_and_phrase_case_insensitive():
    hits = [hit("b.md", "25 days"), hit("a.md", "nothing here"), hit("a.md", "you get 25 days")]
    assert first_relevant_rank(hits, Q) == 3


def test_rank_is_none_when_not_found():
    assert first_relevant_rank([hit("a.md", "no match")], Q) is None


def test_score_metrics():
    s = score([1, 2, None, 1])
    assert s["hit@1"] == 0.5
    assert s["hit@5"] == 0.75
    assert s["mrr"] == (1 + 0.5 + 0 + 1) / 4
