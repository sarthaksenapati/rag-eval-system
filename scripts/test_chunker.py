import sys
sys.path.append(".")
from backend.ingestion.loader import load_from_directory
from backend.ingestion.chunker import chunk_documents, ChunkStrategy

docs = load_from_directory("data/")

for strategy in ChunkStrategy:
    print(f"\n{'='*50}")
    print(f"Strategy: {strategy.value.upper()}")
    print(f"{'='*50}")
    chunks = chunk_documents(docs, strategy)
    print(f"Total chunks: {len(chunks)}")
    print(f"\nFirst chunk preview:")
    print(f"  Length: {len(chunks[0].text)} chars")
    print(f"  Text: {chunks[0].text[:300]}")
    print(f"\nLast chunk preview:")
    print(f"  Length: {len(chunks[-1].text)} chars")
    print(f"  Text: {chunks[-1].text[:300]}")