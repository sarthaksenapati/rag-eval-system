import sys
sys.path.append(".")
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import tempfile, os
from backend.ingestion.loader import load_pdf, load_from_texts
from backend.ingestion.chunker import chunk_documents, ChunkStrategy
from backend.ingestion.embedder import embed_and_upsert, get_collection_count

router = APIRouter()

class IngestTextRequest(BaseModel):
    texts: list[dict]  # [{"text": "...", "source": "..."}]
    strategy: str = "semantic"

class IngestResponse(BaseModel):
    chunks_added: int
    total_in_collection: int
    strategy: str

@router.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    strategy: str = "fixed"
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = load_pdf(tmp_path)
        chunks = chunk_documents(docs, ChunkStrategy(strategy))
        embed_and_upsert(chunks)
    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        chunks_added=len(chunks),
        total_in_collection=get_collection_count(),
        strategy=strategy
    )

@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(req: IngestTextRequest):
    docs = load_from_texts(req.texts)
    chunks = chunk_documents(docs, ChunkStrategy(req.strategy))
    embed_and_upsert(chunks)

    return IngestResponse(
        chunks_added=len(chunks),
        total_in_collection=get_collection_count(),
        strategy=req.strategy
    )

@router.get("/ingest/status")
async def ingest_status():
    return {"total_chunks": get_collection_count()}