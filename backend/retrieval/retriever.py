import sys
sys.path.append(".")
from qdrant_client import QdrantClient
from backend.config import settings

qdrant = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
    timeout=60
)

# Lazy load — model loads on first request not at startup
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embed_model

def search(query: str, top_k: int = 10) -> list[dict]:
    embed_model = get_embed_model()
    query_vector = embed_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = qdrant.search(
        collection_name=settings.collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )

    return [
        {
            "text": r.payload["text"],
            "score": round(r.score, 4),
            "source": r.payload.get("source", "unknown"),
            "page": r.payload.get("page", 0),
            "strategy": r.payload.get("strategy", "unknown"),
        }
        for r in results
    ]