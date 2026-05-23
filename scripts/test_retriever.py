import sys
sys.path.append(".")
from backend.retrieval.retriever import search

# Try a few different queries against your indexed documents
queries = [
    "What does the book say about concealing your intentions?",
    "How should you deal with enemies according to the laws?",
    "What is the law about crushing your enemy totally?",
    "How do you use absence to increase respect?",
    "What does the book say about court politics and power?",
]

for query in queries:
    print(f"\n{'='*55}")
    print(f"Query: {query}")
    print(f"{'='*55}")
    results = search(query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\nResult {i} (score: {r['score']})")
        print(f"  Source: {r['source']} | Page: {r['page']}")
        print(f"  Text: {r['text'][:200]}...")