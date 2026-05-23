import sys
sys.path.append(".")
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from backend.config import settings

qdrant = QdrantClient(url=settings.qdrant_url)
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def search(query: str, top_k: int = 10) -> list[dict]:
    """
    Embeds the query and retrieves the top_k most similar chunks from Qdrant.
    Returns a list of dicts with text, score, and metadata.
    """
    # Embed the query the same way we embedded the chunks
    query_vector = embed_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = qdrant.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

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