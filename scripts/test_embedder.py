import sys
sys.path.append(".")
from backend.ingestion.loader import load_from_directory
from backend.ingestion.chunker import chunk_documents, ChunkStrategy
from backend.ingestion.embedder import create_collection, embed_and_upsert, get_collection_count

docs = load_from_directory("data/")
print(f"Loaded {len(docs)} pages")

chunks = chunk_documents(docs, ChunkStrategy.SEMANTIC)

create_collection(recreate=True)
embed_and_upsert(chunks)

print(f"\nFinal count in Qdrant Cloud: {get_collection_count()}")