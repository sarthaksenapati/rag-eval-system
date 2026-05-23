import sys, asyncio, json
sys.path.append(".")
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.retrieval.retriever import search
from backend.retrieval.reranker import rerank
from backend.generation.llm import generate_answer
from backend.evaluation.ragas_eval import run_ragas_eval

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    top_k: int = 4
    evaluate: bool = False  # set True to run RAGAS after response

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 1. Retrieve
    candidates = search(req.query, top_k=10)

    # 2. Rerank
    reranked = rerank(req.query, candidates, top_k=req.top_k)

    # 3. Generate
    answer = await generate_answer(req.query, reranked)

    # 4. Sources to return to frontend
    sources = [
        {
            "text": r["text"][:300],
            "source": r["source"],
            "page": r["page"],
            "score": r["rerank_score"]
        }
        for r in reranked
    ]

    # 5. Fire and forget eval — doesn't block the response
    if req.evaluate:
        contexts = [r["text"] for r in reranked]
        asyncio.create_task(
            run_eval_async(req.query, answer, contexts)
        )

    return ChatResponse(query=req.query, answer=answer, sources=sources)


async def run_eval_async(query: str, answer: str, contexts: list[str]):
    """Runs RAGAS eval in background without blocking the chat response."""
    try:
        scores = run_ragas_eval(query, answer, contexts)
        print(f"\n[BACKGROUND EVAL] {scores}")
    except Exception as e:
        print(f"[BACKGROUND EVAL ERROR] {e}")


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming version — sends tokens as they arrive."""
    async def generate():
        candidates = search(req.query, top_k=10)
        reranked = rerank(req.query, candidates, top_k=req.top_k)

        # Send sources first
        sources = [
            {"source": r["source"], "page": r["page"], "score": r["rerank_score"]}
            for r in reranked
        ]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Stream answer token by token
        answer = await generate_answer(req.query, reranked)
        for word in answer.split(" "):
            yield f"data: {json.dumps({'type': 'token', 'token': word + ' '})}\n\n"
            await asyncio.sleep(0.02)  # simulate streaming

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")