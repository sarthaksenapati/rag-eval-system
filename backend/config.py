from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    groq_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    database_url: str
    collection_name: str = "rag_docs"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    llm_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    ragas_llm_model: str = "llama-3.3-70b-versatile" 
    hf_token: str = ""# groq model for evals

    class Config:
        env_file = ".env"

settings = Settings()