import sys
sys.path.append(".")
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from backend.retrieval.retriever import search
from backend.retrieval.reranker import rerank
from backend.generation.llm import generate_answer
from backend.config import settings
import httpx, json

# ── State ────────────────────────────────────────────────
class RAGState(TypedDict):
    query: str
    query_type: Literal["simple", "complex"]
    candidates: list
    reranked: list
    answer: str

# ── Nodes ────────────────────────────────────────────────
async def classify_query(state: RAGState) -> RAGState:
    """Uses Groq LLM to classify query as simple or complex."""
    prompt = f"""You are a query classifier for a RAG system.

Classify the query below as "simple" or "complex".

SIMPLE = single factual lookup, one concept, short answer expected
Examples:
- "What is Law 16?"
- "What does the book say about enemies?"

COMPLEX = requires comparing multiple concepts, synthesis, or multi-part reasoning
Examples:
- "Compare how absence and honesty are used as power tactics"
- "How does the book treat friendship vs enemies differently?"

Query: "{state["query"]}"

Reply with exactly one word: simple or complex"""

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 5,
                "temperature": 0,
            }
        )
        result = response.json()
        classification = result["choices"][0]["message"]["content"].strip().lower()

    query_type = "complex" if "complex" in classification else "simple"
    print(f"[CLASSIFIER] '{state['query'][:50]}' → {query_type}")
    return {**state, "query_type": query_type}


async def simple_retrieve(state: RAGState) -> RAGState:
    """Simple path — retrieve top 3, skip reranking."""
    candidates = await search(state["query"], top_k=5)
    top_3 = candidates[:3]
    return {**state, "candidates": candidates, "reranked": top_3}


async def complex_retrieve(state: RAGState) -> RAGState:
    """Complex path — retrieve top 20, full reranking pipeline."""
    candidates = await search(state["query"], top_k=20)
    reranked = rerank(state["query"], candidates, top_k=5)
    return {**state, "candidates": candidates, "reranked": reranked}


async def generate(state: RAGState) -> RAGState:
    """Generate answer from retrieved chunks."""
    answer = await generate_answer(state["query"], state["reranked"])
    return {**state, "answer": answer}


def route_query(state: RAGState) -> Literal["simple_retrieve", "complex_retrieve"]:
    """Conditional edge — routes based on LLM classification."""
    return "simple_retrieve" if state["query_type"] == "simple" else "complex_retrieve"


# ── Build graph ───────────────────────────────────────────
def build_rag_agent():
    graph = StateGraph(RAGState)

    graph.add_node("classify", classify_query)
    graph.add_node("simple_retrieve", simple_retrieve)
    graph.add_node("complex_retrieve", complex_retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_query,
        {
            "simple_retrieve": "simple_retrieve",
            "complex_retrieve": "complex_retrieve"
        }
    )

    graph.add_edge("simple_retrieve", "generate")
    graph.add_edge("complex_retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

rag_agent = build_rag_agent()