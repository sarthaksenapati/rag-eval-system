import sys, asyncio
sys.path.append(".")
from backend.retrieval.retriever import search
from backend.retrieval.reranker import rerank
from backend.generation.llm import generate_answer

async def ask(query: str):
    print(f"\n{'='*55}")
    print(f"Question: {query}")
    print(f"{'='*55}")

    # Retrieve → rerank → generate
    candidates = search(query, top_k=10)
    reranked = rerank(query, candidates, top_k=4)

    print(f"Retrieved {len(candidates)} chunks, reranked to top 4")
    print(f"Top chunk page: {reranked[0]['page']} (reranker score: {reranked[0]['rerank_score']})")
    print(f"\nGenerating answer...\n")

    answer = await generate_answer(query, reranked)
    print(f"Answer:\n{answer}")

async def main():
    queries = [
        "What does the book say about concealing your intentions?",
        "How do you use absence to increase respect and honor?",
        "What is the law about never trusting friends and using enemies?",
    ]
    for query in queries:
        await ask(query)

asyncio.run(main())