import sys, httpx, json
sys.path.append(".")
from backend.config import settings
from backend.generation.prompt_builder import build_prompt, RAGPrompt

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

async def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Builds a RAG prompt from retrieved chunks and calls
    the LLM via OpenRouter. Returns the full answer string.
    """
    rag_prompt = build_prompt(query, chunks)

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
    }

    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": rag_prompt.system},
            {"role": "user",   "content": rag_prompt.user}
        ],
        "temperature": 0.1,   # low temperature — we want factual, grounded answers
        "max_tokens": 1000,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]