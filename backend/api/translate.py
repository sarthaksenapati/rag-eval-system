import sys
sys.path.append(".")
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.config import settings

router = APIRouter()

SUPPORTED_LANGUAGES = {
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "te-IN": "Telugu",
    "mr-IN": "Marathi",
    "ta-IN": "Tamil",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "pa-IN": "Punjabi",
    "or-IN": "Odia",
    "as-IN": "Assamese",
    "ur-IN": "Urdu",
}

class MultilingualChatRequest(BaseModel):
    query: str
    source_language: str = "hi-IN"  # default Hindi
    top_k: int = 4
    translate_response: bool = True  # optionally translate answer back

class MultilingualChatResponse(BaseModel):
    original_query: str
    translated_query: str
    answer: str
    translated_answer: str | None
    sources: list[dict]
    source_language: str

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using Sarvam API."""
    from sarvamai import SarvamAI
    client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
    response = client.text.translate(
        input=text,
        source_language_code=source_lang,
        target_language_code=target_lang,
        model="sarvam-translate:v1"
    )
    return response.translated_text

@router.post("/chat/multilingual", response_model=MultilingualChatResponse)
async def multilingual_chat(req: MultilingualChatRequest):
    """
    Accept queries in any of 22 Indian languages.
    Translates to English, runs RAG pipeline, returns answer.
    Optionally translates answer back to source language.
    """
    from backend.retrieval.retriever import search
    from backend.retrieval.reranker import rerank
    from backend.generation.llm import generate_answer

    if req.source_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    # Step 1 — translate query to English
    if req.source_language == "en-IN":
        translated_query = req.query
    else:
        translated_query = translate_text(
            text=req.query,
            source_lang=req.source_language,
            target_lang="en-IN"
        )

    # Step 2 — run RAG pipeline with English query
    candidates = await search(translated_query, top_k=10)
    reranked = rerank(translated_query, candidates, top_k=req.top_k)
    answer = await generate_answer(translated_query, reranked)

    # Step 3 — optionally translate answer back
    translated_answer = None
    if req.translate_response and req.source_language != "en-IN":
        translated_answer = translate_text(
            text=answer,
            source_lang="en-IN",
            target_lang=req.source_language
        )

    sources = [
        {
            "text": r["text"][:300],
            "source": r["source"],
            "page": r["page"],
            "score": r["rerank_score"]
        }
        for r in reranked
    ]

    return MultilingualChatResponse(
        original_query=req.query,
        translated_query=translated_query,
        answer=answer,
        translated_answer=translated_answer,
        sources=sources,
        source_language=req.source_language
    )