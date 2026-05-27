import sys
import asyncio
sys.path.append(".")
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer
from backend.config import settings

_qdrant_client = None
_embedding_model = None

def get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
            timeout=60
        )
    return _qdrant_client

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model

async def get_embedding(text: str) -> list[float]:
    """Get single embedding using local sentence-transformers model."""
    model = get_embedding_model()
    loop = asyncio.get_event_loop()
    vector = await loop.run_in_executor(None, lambda: model.encode(text).tolist())
    return vector

async def search(query: str, top_k: int = 10) -> list[dict]:
    query_vector = await get_embedding(query)

    results = (await get_qdrant().query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )).points

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