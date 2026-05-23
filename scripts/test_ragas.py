import sys, asyncio, math
sys.path.append(".")
from backend.retrieval.retriever import search
from backend.retrieval.reranker import rerank
from backend.generation.llm import generate_answer
from backend.evaluation.ragas_eval import run_ragas_eval

async def evaluate_query(query: str):
    print(f"\n{'='*55}")
    print(f"Evaluating: {query}")
    print(f"{'='*55}")

    candidates = search(query, top_k=10)
    reranked = rerank(query, candidates, top_k=4)
    answer = await generate_answer(query, reranked)
    contexts = [r["text"] for r in reranked]

    print(f"Answer: {answer[:300]}...")

    scores = run_ragas_eval(
        query=query,
        answer=answer,
        contexts=contexts,
        strategy="semantic"
    )
    return scores

async def main():
    queries = [
        "What does the book say about concealing your intentions?",
        "How do you use absence to increase respect and honor?",
        "What is the law about never trusting friends?",
        "How should you crush your enemies according to the laws?",
        "What does the book say about court politics?",
        "How do you use selective honesty to disarm people?",
        "What is the law about getting others to do the work for you?",
        "How should you guard your reputation according to the book?",
        "What does the book say about playing on people's need to believe?",
        "How do you enter actions with boldness according to the laws?",
    ]

    all_scores = []
    for query in queries:
        scores = await evaluate_query(query)
        all_scores.append(scores)
        await asyncio.sleep(10)  # 10 second pause between queries


    # Print summary table
    print(f"\n{'='*55}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*55}")
    print(f"{'Query':<42} {'Faith':>6} {'Relev':>6}")
    print(f"{'-'*56}")
    for s in all_scores:
        short_q = s['query'][:40] + ".." if len(s['query']) > 40 else s['query']
        faith = f"{s['faithfulness']:>6.3f}" if not math.isnan(s['faithfulness']) else "   nan"
        relev = f"{s['answer_relevancy']:>6.3f}" if not math.isnan(s['answer_relevancy']) else "   nan"
        print(f"{short_q:<42} {faith}  {relev}")

    # Average skipping nan values
    valid_faith = [s['faithfulness'] for s in all_scores if not math.isnan(s['faithfulness'])]
    valid_relev = [s['answer_relevancy'] for s in all_scores if not math.isnan(s['answer_relevancy'])]

    avg_faith = sum(valid_faith) / len(valid_faith) if valid_faith else float('nan')
    avg_relev = sum(valid_relev) / len(valid_relev) if valid_relev else float('nan')

    print(f"\n{'AVERAGE':<42} {avg_faith:>6.3f}  {avg_relev:>6.3f}")
    print(f"{'(valid samples)':<42} {len(valid_faith):>4}/{len(all_scores)}  {len(valid_relev):>3}/{len(all_scores)}")

asyncio.run(main()) 