import sys
sys.path.append(".")
import streamlit as st
import pandas as pd
import requests
import json

API_URL = "http://backend:8000/api"

st.set_page_config(
    page_title="RAG Eval Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG Evaluation Dashboard")
st.caption("Production RAG system with RAGAS evaluation — 48 Laws of Power")

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("System Status")
    try:
        res = requests.get(f"{API_URL}/ingest/status", timeout=3)
        count = res.json()["total_chunks"]
        st.success(f"✅ Qdrant connected")
        st.metric("Chunks in collection", count)
    except:
        st.error("❌ API not reachable — is uvicorn running?")

    st.divider()
    st.header("Settings")
    top_k = st.slider("Top-K chunks", min_value=2, max_value=8, value=4)
    run_eval = st.toggle("Run RAGAS eval", value=False)
    st.caption("Eval adds ~30s per query (Groq rate limits)")

# ── Chat interface ────────────────────────────────────────
st.header("Ask a Question")

query = st.text_input(
    "Query",
    placeholder="What does the book say about concealing your intentions?",
    label_visibility="collapsed"
)

if st.button("Ask", type="primary") and query:
    with st.spinner("Retrieving and generating..."):
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={"query": query, "top_k": top_k, "evaluate": run_eval},
                timeout=60
            )
            data = response.json()

            # Answer
            st.subheader("Answer")
            st.markdown(data["answer"])

            # Sources
            st.subheader("Retrieved Sources")
            for i, src in enumerate(data["sources"], 1):
                with st.expander(f"Source {i} — Page {src['page']} (reranker score: {src['score']})"):
                    st.caption(f"File: {src['source']}")
                    st.write(src["text"])

        except Exception as e:
            st.error(f"Error: {e}")

# ── Batch evaluation ─────────────────────────────────────
st.divider()
st.header("Batch Evaluation")
st.caption("Run multiple queries at once and compare scores")

default_queries = """What does the book say about concealing your intentions?
How do you use absence to increase respect and honor?
What is the law about never trusting friends?
How should you crush your enemies according to the laws?
What does the book say about court politics?"""

batch_input = st.text_area(
    "Enter queries (one per line)",
    value=default_queries,
    height=150
)

if st.button("Run Batch Eval", type="secondary"):
    queries = [q.strip() for q in batch_input.strip().split("\n") if q.strip()]
    results = []

    progress = st.progress(0, text="Running pipeline...")
    status = st.empty()

    for i, q in enumerate(queries):
        status.text(f"Processing: {q[:60]}...")
        try:
            res = requests.post(
                f"{API_URL}/chat",
                json={"query": q, "top_k": top_k, "evaluate": False},
                timeout=60
            )
            data = res.json()
            results.append({
                "Query": q[:60],
                "Answer preview": data["answer"][:120] + "...",
                "Top source page": data["sources"][0]["page"] if data["sources"] else "N/A",
                "Top reranker score": data["sources"][0]["score"] if data["sources"] else 0,
            })
        except Exception as e:
            results.append({
                "Query": q[:60],
                "Answer preview": f"Error: {e}",
                "Top source page": "N/A",
                "Top reranker score": 0,
            })
        progress.progress((i + 1) / len(queries), text=f"Done {i+1}/{len(queries)}")

    status.empty()
    st.subheader("Results")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

# ── RAGAS scores history ──────────────────────────────────
st.divider()
st.header("RAGAS Scores — From eval suite")
st.caption("Paste your eval results from scripts/test_ragas.py here")

scores_data = {
    "Query": [
        "Concealing intentions",
        "Absence and respect",
        "Never trust friends",
        "Crush your enemies",
        "Court politics",
        "Selective honesty",
        "Get others to do the work",
        "Guard your reputation",
    ],
    "Faithfulness": [1.0, 1.0, 0.667, 1.0, 1.0, 1.0, 0.857, None],
    "Answer Relevancy": [0.628, 0.980, 0.809, 0.766, 0.916, 0.872, 0.784, 0.813],
}

df_scores = pd.DataFrame(scores_data)

col1, col2, col3 = st.columns(3)
valid_faith = [x for x in df_scores["Faithfulness"] if x is not None]
valid_relev = [x for x in df_scores["Answer Relevancy"] if x is not None]

col1.metric("Avg Faithfulness", f"{sum(valid_faith)/len(valid_faith):.3f}")
col2.metric("Avg Answer Relevancy", f"{sum(valid_relev)/len(valid_relev):.3f}")
col3.metric("Queries Evaluated", f"{len(valid_faith)}/10")

st.dataframe(df_scores, use_container_width=True)
st.bar_chart(df_scores.set_index("Query")[["Faithfulness", "Answer Relevancy"]])