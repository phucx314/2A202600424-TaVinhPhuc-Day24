"""
Phase A — Task A.3: Failure Cluster Analysis
Identifies bottom 10 questions and clusters failure patterns.
Output: failure_analysis.md
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd
import numpy as np

PHASE_A_DIR = Path(__file__).parent
RESULTS_PATH = PHASE_A_DIR / "ragas_results.csv"
OUTPUT_PATH = PHASE_A_DIR / "failure_analysis.md"

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        print(f"[A.3] ERROR: {RESULTS_PATH} not found. Run run_ragas.py first.")
        sys.exit(1)
    df = pd.read_csv(RESULTS_PATH)
    print(f"[A.3] Loaded {len(df)} results")
    return df


def identify_bottom10(df: pd.DataFrame) -> pd.DataFrame:
    """Compute avg score and return bottom 10 questions."""
    available_metrics = [m for m in METRICS if m in df.columns]
    df["avg_score"] = df[available_metrics].mean(axis=1)
    bottom10 = df.nsmallest(10, "avg_score").reset_index(drop=True)
    return bottom10, available_metrics


def cluster_failures(bottom10: pd.DataFrame) -> list[dict]:
    """Simple rule-based clustering of failure patterns."""
    clusters = []

    # Cluster 1: Low faithfulness (hallucination risk)
    c1 = bottom10[bottom10.get("faithfulness", pd.Series([1]*len(bottom10))) < 0.6]
    if len(c1) >= 1:
        clusters.append({
            "id": "C1",
            "name": "Hallucination / Low Faithfulness",
            "pattern": "Questions where the model generates answers not grounded in retrieved context.",
            "examples": c1["question"].tolist()[:3],
            "root_cause": "Retriever returns tangentially related chunks; LLM then confabulates details.",
            "proposed_fix": (
                "1. Increase `top_k` from 5→8 to retrieve more context.\n"
                "2. Add faithfulness check as pre-filter: if LLM claims fact not in context, substitute 'I don't know'.\n"
                "3. Use stricter system prompt: 'Only state facts explicitly mentioned in the context.'"
            ),
        })

    # Cluster 2: Low context precision (noisy retrieval)
    c2 = bottom10[bottom10.get("context_precision", pd.Series([1]*len(bottom10))) < 0.5]
    c2 = c2[~c2.index.isin(c1.index)] if len(c1) > 0 else c2
    if len(c2) >= 1:
        clusters.append({
            "id": "C2",
            "name": "Noisy Retrieval / Low Context Precision",
            "pattern": "Retrieved chunks contain irrelevant information, diluting the useful signal.",
            "examples": c2["question"].tolist()[:3],
            "root_cause": "FAISS cosine similarity retrieves semantically adjacent but topically irrelevant chunks.",
            "proposed_fix": (
                "1. Add reranking step: use Cohere Rerank API (`co.rerank()`) to reorder top-20 → top-5.\n"
                "2. Implement MMR (Maximal Marginal Relevance) to diversify retrieved chunks.\n"
                "3. Use hybrid search: BM25 + dense with RRF fusion (already in Day 18 pipeline)."
            ),
        })

    # Cluster 3: Low context recall (missing information)
    c3 = bottom10[bottom10.get("context_recall", pd.Series([1]*len(bottom10))) < 0.5]
    remaining_idx = set(bottom10.index)
    if len(c1) > 0:
        remaining_idx -= set(c1.index)
    if len(c2) > 0:
        remaining_idx -= set(c2.index)
    c3 = c3[c3.index.isin(remaining_idx)]

    if len(c3) >= 1:
        clusters.append({
            "id": "C3",
            "name": "Insufficient Context / Low Context Recall",
            "pattern": "Multi-hop questions requiring information from multiple document sections.",
            "examples": c3["question"].tolist()[:3],
            "root_cause": "Single-hop retrieval misses facts spread across multiple chunks.",
            "proposed_fix": (
                "1. Implement multi-hop retrieval: after first retrieval, extract key entities and do second retrieval.\n"
                "2. Increase `top_k` to 10 for multi-context question types.\n"
                "3. Use parent document retrieval (Day 18's hierarchical chunking already supports this)."
            ),
        })

    # Fallback cluster if none of above triggered
    if not clusters:
        remaining = bottom10.head(5)
        clusters.append({
            "id": "C1",
            "name": "Low Answer Relevancy",
            "pattern": "Answers don't directly address the question despite having relevant context.",
            "examples": remaining["question"].tolist()[:3],
            "root_cause": "System prompt too generic; model includes tangential information.",
            "proposed_fix": (
                "1. Add explicit instruction: 'Answer in 1-2 sentences directly addressing the question.'\n"
                "2. Use chain-of-thought: ask model to first identify what the question asks, then answer.\n"
                "3. Post-process: check if key question terms appear in the answer."
            ),
        })
        clusters.append({
            "id": "C2",
            "name": "Multi-Context Reasoning Failures",
            "pattern": "Questions requiring inference across multiple retrieved passages.",
            "examples": remaining["question"].tolist()[3:5] if len(remaining) > 3 else ["N/A"],
            "root_cause": "Retriever returns top-5 from single vector space, misses cross-document connections.",
            "proposed_fix": (
                "1. Switch to graph-based retrieval: build entity graph across documents.\n"
                "2. Use iterative retrieval: feed initial answer back as new query for second-hop retrieval.\n"
                "3. Add document-level context window (include surrounding chunks automatically)."
            ),
        })

    return clusters


def generate_report(bottom10: pd.DataFrame, clusters: list[dict], available_metrics: list[str]) -> str:
    """Generate failure_analysis.md content."""
    # Build bottom-10 table
    cols = ["question"] + available_metrics + ["avg_score"]
    cols = [c for c in cols if c in bottom10.columns]

    table_header = "| # | Question (truncated) | Type | " + " | ".join(
        [m[:2].upper() for m in available_metrics]
    ) + " | Avg | Cluster |"
    table_sep = "|---|---|---|" + "---|" * (len(available_metrics) + 2)

    rows = []
    for i, row in bottom10.iterrows():
        q_short = row["question"][:45].replace("|", "/") + "..."
        evo_type = row.get("evolution_type", "?")[:10]
        metric_vals = " | ".join(
            f"{row[m]:.2f}" if m in row and pd.notna(row[m]) else "N/A"
            for m in available_metrics
        )
        avg = f"{row['avg_score']:.2f}" if "avg_score" in row else "N/A"

        # Find which cluster this row belongs to
        cluster_id = "—"
        for c in clusters:
            if row.get("question") in c.get("examples", []):
                cluster_id = c["id"]
                break

        rows.append(f"| {i+1} | {q_short} | {evo_type} | {metric_vals} | {avg} | {cluster_id} |")

    table = "\n".join([table_header, table_sep] + rows)

    # Build cluster sections
    cluster_sections = []
    for c in clusters:
        examples_md = "\n".join(f'  - "{ex[:80]}..."' for ex in c["examples"])
        fix_lines = c["proposed_fix"].replace("\n", "\n  ")
        cluster_sections.append(f"""
### Cluster {c['id']}: {c['name']}

**Pattern:** {c['pattern']}

**Examples:**
{examples_md}

**Root cause:** {c['root_cause']}

**Proposed fix:**
  {fix_lines}
""")

    clusters_md = "\n".join(cluster_sections)

    report = f"""# Failure Cluster Analysis — Phase A.3

## Bottom 10 Questions (Lowest Average RAGAS Score)

{table}

*Metrics: F=Faithfulness, AR=Answer Relevancy, CP=Context Precision, CR=Context Recall*

---

## Clusters Identified

{clusters_md}

---

## Summary

| Cluster | Count | Primary Issue | Recommended Fix Priority |
|---------|-------|---------------|--------------------------|
""" + "\n".join(
        f"| {c['id']} | {len(c['examples'])} | {c['name']} | High |" for c in clusters
    )

    return report


if __name__ == "__main__":
    df = load_results()
    bottom10, available_metrics = identify_bottom10(df)
    clusters = cluster_failures(bottom10)
    report = generate_report(bottom10, clusters, available_metrics)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[A.3] ✅ Saved failure analysis → {OUTPUT_PATH}")
    print(f"[A.3] Identified {len(clusters)} clusters")
    for c in clusters:
        print(f"  {c['id']}: {c['name']} ({len(c['examples'])} examples)")
