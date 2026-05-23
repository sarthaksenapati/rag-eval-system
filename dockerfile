FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# CPU-only torch — 200MB instead of 2GB, no CUDA needed
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Core ML dependencies
RUN pip install --no-cache-dir transformers accelerate packaging sentence-transformers

# Rest of requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY eval_dashboard/ ./eval_dashboard/
COPY data/ ./data/
COPY scripts/ ./scripts/

ARG HF_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# Download models at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]