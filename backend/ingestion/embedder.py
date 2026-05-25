import sys, time
sys.path.append(".")
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)
from backend.ingestion.chunker import Chunk
from backend.config import settings
import uuid

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embed_model

qdrant = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
    timeout=60
)

def create_collection(recreate: bool = False):
    existing = [c.name for c in qdrant.get_collections().collections]
    if settings.collection_name in existing:
        if recreate:
            qdrant.delete_collection(settings.collection_name)
            print(f"Deleted existing collection: {settings.collection_name}")
        else:
            print(f"Collection already exists: {settings.collection_name}")
            return
    qdrant.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(
            size=settings.embedding_dim,
            distance=Distance.COSINE
        )
    )
    print(f"Created collection: {settings.collection_name}")

def embed_and_upsert(chunks: list[Chunk], batch_size: int = 20):
    print(f"Embedding and upserting {len(chunks)} chunks...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]

        vectors = get_embed_model().encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        ).tolist()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": chunk.metadata.get("file_name", "unknown"),
                    "page": chunk.metadata.get("page", 0),
                    "strategy": chunk.strategy,
                    "chunk_id": chunk.chunk_id,
                }
            )
            for chunk, vector in zip(batch, vectors)
        ]

        for attempt in range(3):
            try:
                qdrant.upsert(
                    collection_name=settings.collection_name,
                    points=points
                )
                print(f"Upserted batch {i // batch_size + 1} ({len(batch)} chunks)")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"Batch {i // batch_size + 1} failed, retrying ({attempt + 1}/3)...")
                    time.sleep(5)
                else:
                    print(f"Batch {i // batch_size + 1} failed after 3 attempts: {e}")

    print(f"\nDone. Total chunks in Qdrant: {get_collection_count()}")

def get_collection_count() -> int:
    try:
        info = qdrant.get_collection(settings.collection_name)
        return info.points_count
    except Exception:
        return 0