import sys
sys.path.append(".")
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from backend.config import settings

def get_ragas_llm():
    return LangchainLLMWrapper(ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=2000,
    ))

def get_ragas_embeddings():
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    ))

def run_ragas_eval(
    query: str,
    answer: str,
    contexts: list[str],
    strategy: str = "semantic"
) -> dict:
    dataset = Dataset.from_dict({
        "question": [query],
        "answer":   [answer],
        "contexts": [contexts],
    })

    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    metrics = [faithfulness, answer_relevancy]
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings
            
    answer_relevancy.strictness = 1

    run_config = RunConfig(
        timeout=None,
        max_retries=1,      # only 1 retry
        max_wait=10,
        max_workers=1,      # sequential — no parallel calls
    )

    result = evaluate(
        dataset,
        metrics=metrics,
        run_config=run_config,
    )
    scores = result.to_pandas().iloc[0].to_dict()

    output = {
        "query": query,
        "strategy": strategy,
        "faithfulness": round(float(scores.get("faithfulness", 0)), 4),
        "answer_relevancy": round(float(scores.get("answer_relevancy", 0)), 4),
    }

    print(f"\nRAGAS scores:")
    print(f"  Faithfulness     : {output['faithfulness']}")
    print(f"  Answer relevancy : {output['answer_relevancy']}")

    return output