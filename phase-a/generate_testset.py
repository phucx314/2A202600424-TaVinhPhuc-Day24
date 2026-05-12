"""
Phase A — Task A.1: Synthetic Test Set Generation
Uses OpenAI directly to generate 50 questions with correct distribution.
(RAGAS 0.4.x TestsetGenerator is incompatible with Python 3.14 + plain-text corpus)
Output: phase-a/testset_v1.csv
"""

import os
import sys
import json
import random
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd
from openai import OpenAI
from pypdf import PdfReader

OUTPUT_DIR = Path(__file__).parent
DAY18_DATA = Path(__file__).parent.parent.parent / "day018" / "2A202600424-TaVinhPhuc-Day18" / "data"
DOCS_DIR = Path(__file__).parent.parent / "docs"

client = OpenAI()


# ─────────────────────────────────────────────────────────────────────────────
# Load corpus
# ─────────────────────────────────────────────────────────────────────────────
def load_corpus() -> list[str]:
    """Load all text chunks from Day 18 corpus."""
    chunks = []
    source_dir = DAY18_DATA if DAY18_DATA.exists() else DOCS_DIR
    print(f"[A.1] Loading corpus from: {source_dir}")

    for f in source_dir.glob("**/*.md"):
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) > 200:
            # Split into ~800-char chunks
            for i in range(0, len(text), 800):
                chunk = text[i:i+800].strip()
                if len(chunk) > 100:
                    chunks.append(chunk)

    for f in source_dir.glob("**/*.pdf"):
        try:
            reader = PdfReader(str(f))
            for page in reader.pages:
                text = (page.extract_text() or "").strip()
                if len(text) > 100:
                    chunks.append(text[:1000])
            print(f"  PDF loaded: {f.name} ({len(reader.pages)} pages)")
        except Exception as e:
            print(f"  Skip {f.name}: {e}")

    # Always add these rich domain chunks (ensures good coverage for all question types)
    extra_chunks = [
        "RAG (Retrieval-Augmented Generation) systems combine a retrieval component with a language model. The retrieval step finds relevant document chunks using vector similarity search. The generation step uses an LLM to synthesize an answer grounded in the retrieved context. Key metrics: faithfulness measures if the answer is supported by context; answer relevancy measures if the answer addresses the question; context precision measures retrieval signal-to-noise ratio; context recall measures if all needed information was retrieved.",
        "RAGAS is an open-source framework for evaluating RAG pipelines. It computes four core metrics automatically using an LLM judge: faithfulness (0-1), answer relevancy (0-1), context precision (0-1), context recall (0-1). Target thresholds for production: faithfulness ≥ 0.85, answer relevancy ≥ 0.80, context precision ≥ 0.70, context recall ≥ 0.75. RAGAS also supports synthetic test set generation from document corpora.",
        "LLM-as-Judge uses a language model to evaluate other models' outputs. Pairwise comparison presents two answers and asks which is better. Absolute scoring rates one answer on multiple dimensions (1-5 scale). Known biases: position bias (prefers first answer), length bias (prefers longer answers), self-preference bias (model favors its own outputs). Mitigation: swap-and-average for position bias, explicit rubrics for length.",
        "Cohen's kappa is a statistical measure of inter-rater agreement that corrects for chance. Formula: kappa = (Po - Pe) / (1 - Pe). Interpretation: <0 worse than chance, 0-0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial (production-ready), >0.8 almost perfect. Used in LLM evaluation to calibrate LLM judge against human labels.",
        "Guardrails for LLM systems work at two levels. Input guardrails: PII redaction removes personal information before the LLM sees it; topic scope validation rejects off-domain queries; injection detection blocks prompt injection attacks. Output guardrails: Llama Guard 3 classifies responses by safety category (violence, hate speech, self-harm, illegal activities). A full stack runs input guards in parallel (L1), then LLM generation (L2), then output safety check (L3), with async audit logging (L4).",
        "PII (Personally Identifiable Information) detection uses two techniques: regex patterns for structured formats (phone numbers, ID cards, emails, tax codes) and NER models for unstructured text (names, addresses). For Vietnamese documents, specific patterns are needed: CCCD (12 digits), phone (+84 or 0 prefix, 9-10 digits), tax codes (10 digits with optional suffix). Microsoft Presidio provides multilingual NER. Target: recall ≥ 80%, latency P95 < 50ms.",
        "Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân của Việt Nam quy định: dữ liệu cá nhân cơ bản bao gồm họ tên, ngày sinh, giới tính, nơi sinh, ảnh cá nhân. Dữ liệu cá nhân nhạy cảm bao gồm hồ sơ y tế, dữ liệu sinh trắc học, thông tin tài chính. Tổ chức phải có sự đồng ý của chủ thể trước khi thu thập. Vi phạm sẽ bị xử phạt hành chính.",
        "Latency in multi-layer guardrail stacks must be carefully managed. L1 input guards run in parallel using asyncio.gather() to avoid sequential overhead. Target budgets: L1 (PII + topic) P95 < 50ms, L2 (RAG generation) P95 < 2000ms, L3 (Llama Guard via Groq API) P95 < 100ms. L4 audit logging is fire-and-forget and does not count toward latency budget. Total end-to-end P95 target: < 2.5 seconds.",
        "FAISS (Facebook AI Similarity Search) enables fast approximate nearest neighbor search over dense vector embeddings. IndexFlatIP performs exact inner product search (equivalent to cosine similarity after L2 normalization). FAISS supports billion-scale indices. For RAG, embeddings are pre-computed for all chunks, stored in FAISS index, then query embedding is compared at inference time. Typical latency: 1-5ms for top-5 retrieval over 100k vectors.",
        "Adversarial attacks on LLM systems include: DAN (Do Anything Now) prompts that try to override safety guidelines, role-play attacks that create fictional personas without restrictions, payload splitting that spreads harmful content across multiple messages, encoding attacks that obfuscate harmful instructions in Base64 or ROT13, and indirect injection attacks embedded in retrieved documents. Detection approaches: topic guard (embedding similarity), keyword filtering, and model-based classifiers like Prompt Guard.",
    ]
    chunks.extend(extra_chunks)

    print(f"[A.1] Total corpus: {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Question generators
# ─────────────────────────────────────────────────────────────────────────────
def generate_simple_questions(chunks: list[str], n: int) -> list[dict]:
    """Generate single-hop questions directly answerable from one chunk."""
    print(f"[A.1] Generating {n} simple (single-hop) questions...")
    results = []
    sample = random.sample(chunks, min(n, len(chunks)))

    for i, chunk in enumerate(sample[:n]):
        print(f"  simple [{i+1}/{n}]...")
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a test set generator. Generate one clear, specific question that can be answered directly from the provided text. Output JSON only."},
                    {"role": "user", "content": f"""Text: {chunk[:600]}

Generate ONE question answerable from this text.
Output JSON: {{"question": "...", "ground_truth": "one sentence answer from the text"}}"""},
                ],
            )
            data = json.loads(resp.choices[0].message.content.strip().replace("```json","").replace("```",""))
            results.append({
                "question": data["question"],
                "ground_truth": data["ground_truth"],
                "contexts": [chunk],
                "evolution_type": "simple",
            })
        except Exception as e:
            print(f"    ⚠️  {e}")
            results.append({
                "question": f"What is described in this passage about {chunk[:50]}?",
                "ground_truth": chunk[:200],
                "contexts": [chunk],
                "evolution_type": "simple",
            })

    return results


def generate_reasoning_questions(chunks: list[str], n: int) -> list[dict]:
    """Generate reasoning questions requiring inference."""
    print(f"[A.1] Generating {n} reasoning questions...")
    results = []
    sample = random.sample(chunks, min(n, len(chunks)))

    for i, chunk in enumerate(sample[:n]):
        print(f"  reasoning [{i+1}/{n}]...")
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "Generate one reasoning question that requires multi-step inference from the text, not just direct lookup. Output JSON only."},
                    {"role": "user", "content": f"""Text: {chunk[:600]}

Generate ONE question requiring inference or comparison (not just fact lookup).
Output JSON: {{"question": "...", "ground_truth": "reasoned answer based on the text"}}"""},
                ],
            )
            data = json.loads(resp.choices[0].message.content.strip().replace("```json","").replace("```",""))
            results.append({
                "question": data["question"],
                "ground_truth": data["ground_truth"],
                "contexts": [chunk],
                "evolution_type": "reasoning",
            })
        except Exception as e:
            print(f"    ⚠️  {e}")
            results.append({
                "question": f"Why is the approach described in this passage important?",
                "ground_truth": chunk[:200],
                "contexts": [chunk],
                "evolution_type": "reasoning",
            })

    return results


def generate_multi_context_questions(chunks: list[str], n: int) -> list[dict]:
    """Generate questions requiring info from multiple chunks."""
    print(f"[A.1] Generating {n} multi-context questions...")
    results = []

    for i in range(n):
        print(f"  multi-context [{i+1}/{n}]...")
        # Pick 2 random chunks
        pair = random.sample(chunks, min(2, len(chunks)))
        ctx1, ctx2 = pair[0], pair[-1]

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "Generate one question that requires combining information from BOTH texts to answer. Output JSON only."},
                    {"role": "user", "content": f"""Text 1: {ctx1[:400]}

Text 2: {ctx2[:400]}

Generate ONE question requiring information from BOTH texts.
Output JSON: {{"question": "...", "ground_truth": "answer combining both texts"}}"""},
                ],
            )
            data = json.loads(resp.choices[0].message.content.strip().replace("```json","").replace("```",""))
            results.append({
                "question": data["question"],
                "ground_truth": data["ground_truth"],
                "contexts": [ctx1, ctx2],
                "evolution_type": "multi_context",
            })
        except Exception as e:
            print(f"    ⚠️  {e}")
            results.append({
                "question": "How do these two concepts relate to each other in an AI evaluation system?",
                "ground_truth": f"{ctx1[:100]} ... {ctx2[:100]}",
                "contexts": [ctx1, ctx2],
                "evolution_type": "multi_context",
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def generate_testset(chunks: list[str], test_size: int = 50) -> pd.DataFrame:
    """Generate full testset with 50/25/25 distribution."""
    random.seed(42)
    n_simple = int(test_size * 0.50)        # 25 simple
    n_reasoning = int(test_size * 0.25)     # 13 reasoning
    n_multi = test_size - n_simple - n_reasoning  # 12 multi-context

    all_rows = []
    all_rows.extend(generate_simple_questions(chunks, n_simple))
    all_rows.extend(generate_reasoning_questions(chunks, n_reasoning))
    all_rows.extend(generate_multi_context_questions(chunks, n_multi))

    # Shuffle
    random.shuffle(all_rows)

    df = pd.DataFrame(all_rows)

    # Ensure contexts is stored as string (list → str for CSV)
    df["contexts"] = df["contexts"].apply(lambda x: str(x) if isinstance(x, list) else x)

    out_path = OUTPUT_DIR / "testset_v1.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[A.1] ✅ Saved {len(df)} questions → {out_path}")
    print("\n  Distribution:")
    print(df["evolution_type"].value_counts().to_string())

    required = {"question", "ground_truth", "contexts", "evolution_type"}
    missing = required - set(df.columns)
    if missing:
        print(f"\n  ⚠️  Missing columns: {missing}")
    else:
        print(f"\n  ✅ All 4 required columns present: {sorted(required)}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=50)
    args = parser.parse_args()

    chunks = load_corpus()
    generate_testset(chunks, test_size=args.size)
