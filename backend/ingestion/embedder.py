import sys, asyncio
sys.path.append(".")
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType
)
from sentence_transformers import SentenceTransformer
from backend.ingestion.chunker import Chunk
from backend.config import settings
import uuid

# Load embedding model once at module level — reused across all calls
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
qdrant = QdrantClient(url=settings.qdrant_url)


def create_collection(recreate: bool = False):
    """Creates the Qdrant collection. Set recreate=True to wipe and start fresh."""
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


def embed_and_upsert(chunks: list[Chunk], batch_size: int = 64):
    """Embeds chunks and upserts them into Qdrant in batches."""
    print(f"Embedding and upserting {len(chunks)} chunks...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]

        # Embed the whole batch at once — much faster than one by one
        vectors = embed_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True  # important for cosine similarity
        ).tolist()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),  # fresh UUID for each point
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

        qdrant.upsert(
            collection_name=settings.collection_name,
            points=points
        )
        print(f"Upserted batch {i // batch_size + 1} ({len(batch)} chunks)")

    print(f"\nDone. Total chunks in Qdrant: {get_collection_count()}")


def get_collection_count() -> int:
    try:
        info = qdrant.get_collection(settings.collection_name)
        return info.points_count
    except Exception:
        return 0