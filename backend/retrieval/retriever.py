import sys
import httpx
sys.path.append(".")
from qdrant_client import AsyncQdrantClient
from backend.config import settings

_qdrant_client = None

def get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
            timeout=60
        )
    return _qdrant_client

async def get_embedding(text: str) -> list[float]:
    """Get single embedding via HuggingFace Inference API."""
    if not settings.hf_token:
        raise ValueError("HF_TOKEN environment variable is not set")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5",
            headers={"Authorization": f"Bearer {settings.hf_token}"},
            json={"inputs": [text], "options": {"wait_for_model": True}},
        )
        response.raise_for_status()
        return response.json()[0]

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