from dataclasses import dataclass

@dataclass
class RAGPrompt:
    system: str
    user: str
    context_chunks: list[dict]

SYSTEM_PROMPT = """You are a precise AI assistant. Answer questions using ONLY the provided context.
If the context does not contain enough information to answer, say "I don't have enough information to answer this."
Always cite which source page(s) you used in your answer using [Page N] notation.
Be concise but complete."""

def build_prompt(query: str, chunks: list[dict]) -> RAGPrompt:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        page = chunk.get("page", "?")
        context_parts.append(f"[Page {page}]:\n{chunk['text']}")

    context_str = "\n\n---\n\n".join(context_parts)

    user_message = f"""Context:
{context_str}

Question: {query}

Answer (cite pages):"""

    return RAGPrompt(
        system=SYSTEM_PROMPT,
        user=user_message,
        context_chunks=chunks
    )