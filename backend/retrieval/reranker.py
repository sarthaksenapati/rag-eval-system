import sys
sys.path.append(".")
from sentence_transformers import CrossEncoder

_reranker_model = None

def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512
        )
    return _reranker_model

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