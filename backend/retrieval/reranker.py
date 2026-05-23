import sys
sys.path.append(".")
from sentence_transformers import CrossEncoder

# Downloads once (~85MB), runs locally, no API key needed
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512
)

def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Takes retrieved candidates and reranks them using a cross-encoder.
    Returns top_k results sorted by reranker score.
    """
    if not candidates:
        return []

    # Cross-encoder scores query+chunk together as a pair
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker_model.predict(pairs, show_progress_bar=False)

    # Attach reranker scores
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = round(float(score), 4)

    # Sort by reranker score descending
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]