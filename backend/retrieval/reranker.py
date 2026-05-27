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

