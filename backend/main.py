import sys
sys.path.append(".")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import chat, ingest, translate, agent

app = FastAPI(
    title="RAG Eval System",
    description="Production RAG pipeline with RAGAS evaluation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(ingest.router, prefix="/api", tags=["ingest"])
app.include_router(translate.router, prefix="/api", tags=["multilingual"])
app.include_router(agent.router, prefix="/api", tags=["agent"])


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}