import sys
sys.path.append(".")
from backend.retrieval.retriever import search
from backend.retrieval.reranker import rerank

queries = [
    "What does the book say about concealing your intentions?",
    "How should you deal with enemies according to the laws?",
    "How do you use absence to increase respect?",
]

for query in queries:
    print(f"\n{'='*55}")
    print(f"Query: {query}")
    print(f"{'='*55}")

    # Step 1: retrieve top 10 candidates
    candidates = search(query, top_k=10)

    # Step 2: rerank and keep top 3
    reranked = rerank(query, candidates, top_k=3)

    print(f"\nAfter reranking (top 3):")
    for i, r in enumerate(reranked, 1):
        print(f"\nResult {i}")
        print(f"  Retrieval score : {r['score']}")
        print(f"  Reranker score  : {r['rerank_score']}")
        print(f"  Source: {r['source']} | Page: {r['page']}")
        print(f"  Text: {r['text'][:250]}...")