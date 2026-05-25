import sys
sys.path.append(".")
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from backend.retrieval.retriever import search
from backend.retrieval.reranker import rerank
from backend.generation.llm import generate_answer
from backend.config import settings

# ── State ────────────────────────────────────────────────
class RAGState(TypedDict):
    query: str
    query_type: Literal["simple", "complex"]
    candidates: list
    reranked: list
    answer: str

# ── Nodes ────────────────────────────────────────────────
def classify_query(state: RAGState) -> RAGState:
    """
    Classifies query as simple or complex.
    Simple: single question, short, factual
    Complex: multi-part, comparative, analytical
    """
    query = state["query"].strip()
    complex_signals = [
        " and ", " vs ", " compare", " difference",
        " how does", " why does", " explain",
        "?", "multiple", " both"
    ]
    query_lower = query.lower()
    word_count = len(query.split())

    is_complex = (
        word_count > 12 or
        sum(1 for s in complex_signals if s in query_lower) >= 2
    )

    return {**state, "query_type": "complex" if is_complex else "simple"}


def simple_retrieve(state: RAGState) -> RAGState:
    """
    Simple path — retrieve top 3, skip reranking.
    Faster, used for straightforward factual queries.
    """
    candidates = search(state["query"], top_k=5)
    # No reranking — just take top 3 by embedding score
    top_3 = candidates[:3]
    return {**state, "candidates": candidates, "reranked": top_3}


def complex_retrieve(state: RAGState) -> RAGState:
    """
    Complex path — retrieve top 20, full reranking pipeline.
    More thorough, used for multi-part or analytical queries.
    """
    candidates = search(state["query"], top_k=20)
    reranked = rerank(state["query"], candidates, top_k=5)
    return {**state, "candidates": candidates, "reranked": reranked}


async def generate(state: RAGState) -> RAGState:
    """Generate answer from retrieved chunks."""
    answer = await generate_answer(state["query"], state["reranked"])
    return {**state, "answer": answer}


def route_query(state: RAGState) -> Literal["simple_retrieve", "complex_retrieve"]:
    """Conditional edge — routes based on query classification."""
    return "simple_retrieve" if state["query_type"] == "simple" else "complex_retrieve"


# ── Build graph ───────────────────────────────────────────
def build_rag_agent():
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("classify", classify_query)
    graph.add_node("simple_retrieve", simple_retrieve)
    graph.add_node("complex_retrieve", complex_retrieve)
    graph.add_node("generate", generate)

    # Entry point
    graph.set_entry_point("classify")

    # Conditional edge after classification
    graph.add_conditional_edges(
        "classify",
        route_query,
        {
            "simple_retrieve": "simple_retrieve",
            "complex_retrieve": "complex_retrieve"
        }
    )

    # Both paths lead to generation
    graph.add_edge("simple_retrieve", "generate")
    graph.add_edge("complex_retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

# Single compiled agent instance
rag_agent = build_rag_agent()