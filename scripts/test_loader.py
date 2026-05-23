import sys
sys.path.append(".")
from backend.ingestion.loader import load_from_directory

docs = load_from_directory("data/")

print(f"\nTotal documents loaded: {len(docs)}")
for i, doc in enumerate(docs[:3]):  # preview first 3
    print(f"\n--- Doc {i+1} ---")
    print(f"Source: {doc.metadata.get('file_name', 'unknown')}")
    print(f"Text preview: {doc.text[:200]}...")