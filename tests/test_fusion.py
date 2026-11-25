from app.retrieval import Hit, reciprocal_rank_fusion


def hit(cid: int, score: float = 0.0) -> Hit:
    return Hit(chunk_id=cid, source="doc.md", heading="", content=f"chunk {cid}", score=score)


def test_item_found_by_both_searches_wins():
    vector = [hit(1), hit(2), hit(3)]
    keyword = [hit(3), hit(4)]
    fused = reciprocal_rank_fusion(vector, keyword)
    assert fused[0].chunk_id == 3


def test_raw_scores_do_not_matter_only_rank():
    vector = [hit(1, score=0.99), hit(2, score=0.98)]
    keyword = [hit(2, score=0.001), hit(1, score=0.0005)]
    fused = reciprocal_rank_fusion(vector, keyword)
    assert {h.chunk_id for h in fused} == {1, 2}
    assert fused[0].score == fused[1].score  # symmetric positions give equal scores


def test_each_chunk_appears_once():
    fused = reciprocal_rank_fusion([hit(1), hit(2)], [hit(2), hit(1)], [hit(1)])
    assert [h.chunk_id for h in fused].count(1) == 1
    assert fused[0].chunk_id == 1
