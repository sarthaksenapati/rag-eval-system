import sys, asyncio, json
sys.path.append(".")
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.retrieval.retriever import search
from backend.retrieval.reranker import rerank
from backend.generation.llm import generate_answer

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    top_k: int = 4
    evaluate: bool = False

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    candidates = await search(req.query, top_k=10)
    reranked = rerank(req.query, candidates, top_k=req.top_k)
    answer = await generate_answer(req.query, reranked)

    sources = [
        {
            "text": r["text"][:300],
            "source": r["source"],
            "page": r["page"],
            "score": r["rerank_score"]
        }
        for r in reranked
    ]

    if req.evaluate:
        contexts = [r["text"] for r in reranked]
        asyncio.create_task(
            run_eval_async(req.query, answer, contexts)
        )

    return ChatResponse(query=req.query, answer=answer, sources=sources)


async def run_eval_async(query: str, answer: str, contexts: list[str]):
    try:
        # Lazy import — only loads RAGAS when eval is actually requested
        from backend.evaluation.ragas_eval import run_ragas_eval
        scores = run_ragas_eval(query, answer, contexts)
        print(f"\n[BACKGROUND EVAL] {scores}")
    except Exception as e:
        print(f"[BACKGROUND EVAL ERROR] {e}")


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate():
        candidates = await search(req.query, top_k=10)
        reranked = rerank(req.query, candidates, top_k=req.top_k)

        sources = [
            {"source": r["source"], "page": r["page"], "score": r["rerank_score"]}
            for r in reranked
        ]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        answer = await generate_answer(req.query, reranked)
        for word in answer.split(" "):
            yield f"data: {json.dumps({'type': 'token', 'token': word + ' '})}\n\n"
            await asyncio.sleep(0.02)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")