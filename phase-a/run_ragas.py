"""
Phase A — Task A.2: Run RAGAS 4 Metrics
Evaluates RAG pipeline on testset_v1.csv using all 4 RAGAS metrics.
Output: ragas_results.csv, ragas_summary.json
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd
from langchain_openai import ChatOpenAI

PHASE_A_DIR = Path(__file__).parent
TESTSET_PATH = PHASE_A_DIR / "testset_v1.csv"
RESULTS_PATH = PHASE_A_DIR / "ragas_results.csv"
SUMMARY_PATH = PHASE_A_DIR / "ragas_summary.json"


def load_testset() -> pd.DataFrame:
    if not TESTSET_PATH.exists():
        print(f"[A.2] ERROR: {TESTSET_PATH} not found. Run generate_testset.py first.")
        sys.exit(1)
    df = pd.read_csv(TESTSET_PATH)
    print(f"[A.2] Loaded testset: {len(df)} questions")
    return df


def run_rag_on_testset(df: pd.DataFrame) -> list[dict]:
    """Run RAG pipeline on every question in testset."""
    from rag_adapter import RAGPipeline
    rag = RAGPipeline()

    results = []
    total = len(df)
    for i, row in df.iterrows():
        question = row["question"]
        ground_truth = row.get("ground_truth", "")

        print(f"  [{i+1}/{total}] {question[:60]}...")
        try:
            answer, contexts = rag.query(question)
        except Exception as e:
            print(f"    ⚠️  RAG failed: {e} — using fallback")
            answer = "I don't have enough information."
            contexts = []

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    return results


def evaluate_with_ragas(results: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Run RAGAS evaluation on all 4 metrics (RAGAS 0.4.3 compatible).
    
    Bug found & fixed: RAGAS 0.4.3 answer_relevancy internally calls
    embeddings.embed_query(). The RAGAS-native OpenAIEmbeddings only has
    embed_text(), causing AttributeError. Fix: wrap langchain's OpenAIEmbeddings
    (which has embed_query) in LangchainEmbeddingsWrapper.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from datasets import Dataset
    from langchain_openai import OpenAIEmbeddings

    print("\n[A.2] Initializing judge LLM and embeddings...")

    # LangchainEmbeddingsWrapper exposes embed_query() which RAGAS answer_relevancy needs
    lc_emb = OpenAIEmbeddings(model="text-embedding-3-small")
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_emb)
    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))

    print("[A.2] Running RAGAS evaluation (4 metrics, 2 workers)...")

    # RAGAS 0.4.x expects: user_input, response, retrieved_contexts, reference
    ragas_records = []
    for r in results:
        ctx = r.get("contexts", [])
        if isinstance(ctx, str):
            import ast
            try: ctx = ast.literal_eval(ctx)
            except: ctx = [ctx]
        ragas_records.append({
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": ctx if isinstance(ctx, list) else [str(ctx)],
            "reference": str(r.get("ground_truth", "")),
        })

    dataset = Dataset.from_list(ragas_records)

    run_config = RunConfig(timeout=600, max_workers=2)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    results_df = scores.to_pandas()
    return results_df, scores

def save_results(results_df: pd.DataFrame, scores) -> dict:
    """Save CSV + JSON summary."""
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n[A.2] ✅ Saved results → {RESULTS_PATH}")

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = {}
    for m in metrics:
        try:
            val = float(scores[m])
        except (KeyError, TypeError):
            val = float(results_df[m].mean()) if m in results_df else 0.0
        summary[m] = round(val, 4)

    summary["judge_model"] = "gpt-4o-mini"
    summary["n_questions"] = len(results_df)

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[A.2] ✅ Saved summary → {SUMMARY_PATH}")

    # Print results table
    targets = {"faithfulness": 0.85, "answer_relevancy": 0.80, "context_precision": 0.70, "context_recall": 0.75}
    print("\n" + "=" * 55)
    print("  RAGAS Results")
    print("=" * 55)
    total_cost_note = ""
    for m, v in summary.items():
        if m in metrics:
            target = targets[m]
            status = "✅" if v >= target else ("⚠️ " if v >= target - 0.1 else "❌")
            print(f"  {status} {m:<25} {v:.4f}  (target ≥ {target})")
    print("=" * 55)
    print("\n  💡 Log total API cost manually in README.md")
    print("     Estimate: ~$1-2 for 50 questions with gpt-4o-mini")

    return summary


if __name__ == "__main__":
    df = load_testset()
    # Limit to 10 questions for lab time feasibility
    if len(df) > 10:
        print(f"[A.2] Truncating testset from {len(df)} to 10 questions for faster evaluation.")
        df = df.head(10)
    
    results = run_rag_on_testset(df)
    results_df, scores = evaluate_with_ragas(results)
    summary = save_results(results_df, scores)
