import sys
sys.path.append(".")

def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Returns top_k candidates sorted by retrieval score.
    Cross-encoder reranking disabled on cloud deployment to reduce memory.
    """
    sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    for c in sorted_candidates:
        c["rerank_score"] = c["score"]
    return sorted_candidates[:top_k]

def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []

    reranker_model = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker_model.predict(pairs, show_progress_bar=False)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = round(float(score), 4)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]