import sys
sys.path.append(".")
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class AgentRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    query: str
    query_type: str
    answer: str
    sources: list[dict]
    path_taken: str

@router.post("/agent/chat", response_model=AgentResponse)
async def agent_chat(req: AgentRequest):
    from backend.agent.rag_agent import rag_agent  # lazy import

    initial_state = {
        "query": req.query,
        "query_type": "simple",
        "candidates": [],
        "reranked": [],
        "answer": ""
    }

    result = await rag_agent.ainvoke(initial_state)

    sources = [
        {
            "text": r["text"][:300],
            "source": r["source"],
            "page": r["page"],
            "score": r.get("rerank_score", r.get("score", 0))
        }
        for r in result["reranked"]
    ]

    path = "simple (no reranking)" if result["query_type"] == "simple" else "complex (full reranking)"

    return AgentResponse(
        query=req.query,
        query_type=result["query_type"],
        answer=result["answer"],
        sources=sources,
        path_taken=path
    )