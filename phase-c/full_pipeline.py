"""
Phase C — Task C.5: Full Stack Integration & Latency Benchmark
4-layer guardrail pipeline with async parallel execution.
Benchmark: ≥ 100 requests, reports P50/P95/P99.

Run: python phase-c/full_pipeline.py
"""

import os
import sys
import time
import asyncio
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
import pandas as pd

PHASE_C_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# Audit Logger (L4 — fire-and-forget async)
# ─────────────────────────────────────────────────────────────────────────────
AUDIT_LOG_PATH = PHASE_C_DIR / "audit_log.jsonl"

async def audit_log(user_input: str, response: str, timings: dict, blocked: bool = False):
    """Async audit logging — does not count toward latency budget."""
    import json
    record = {
        "timestamp": time.time(),
        "user_input_preview": user_input[:80],
        "response_preview": response[:80] if response else "(blocked)",
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "blocked": blocked,
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def refuse_response(reason: str = "off-topic") -> str:
    return (
        f"Xin lỗi, tôi không thể xử lý yêu cầu này ({reason}). "
        "Vui lòng đặt câu hỏi trong phạm vi hỗ trợ của hệ thống."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full Guarded Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class GuardedPipeline:
    """
    4-layer defense-in-depth guardrail stack:
    L1: Input (PII + Topic) — parallel
    L2: RAG LLM (Day 18 pipeline)
    L3: Output (Llama Guard) — async
    L4: Audit log — fire-and-forget
    """

    def __init__(self, allowed_topics: list[str] | None = None):
        from phase_c.input_guard import InputGuard, TopicGuard, ALLOWED_TOPICS
        from phase_c.output_guard import OutputGuard
        from rag_adapter import RAGPipeline

        self.input_guard = InputGuard()
        self.topic_guard = TopicGuard(
            allowed_topics or ALLOWED_TOPICS,
            threshold=0.50,
        )
        self.output_guard = OutputGuard()
        self.rag = RAGPipeline()
        print("[Pipeline] All layers initialized ✅")

    async def run(self, user_input: str) -> tuple[str, dict]:
        """
        Run the full 4-layer pipeline.
        Returns: (response, timings_dict)
        """
        timings = {}

        # ── L1: Input guardrails (parallel) ────────────────────
        t0 = time.perf_counter()
        pii_task = asyncio.create_task(self.input_guard.sanitize_async(user_input))
        topic_task = asyncio.create_task(self.topic_guard.check_async(user_input))

        (sanitized, _), (topic_ok, topic_reason) = await asyncio.gather(pii_task, topic_task)
        timings["L1"] = (time.perf_counter() - t0) * 1000

        if not topic_ok:
            response = refuse_response("off-topic")
            asyncio.create_task(audit_log(user_input, response, timings, blocked=True))
            return response, timings

        # ── L2: RAG LLM ────────────────────────────────────────
        t0 = time.perf_counter()
        answer, contexts = await self.rag.query_async(sanitized)
        timings["L2"] = (time.perf_counter() - t0) * 1000

        # ── L3: Output safety (async) ───────────────────────────
        t0 = time.perf_counter()
        is_safe, guard_result, guard_latency = await self.output_guard.check_async(
            sanitized, answer
        )
        timings["L3"] = (time.perf_counter() - t0) * 1000

        if not is_safe:
            response = refuse_response("unsafe content detected")
            asyncio.create_task(audit_log(user_input, response, timings, blocked=True))
            return response, timings

        # ── L4: Audit log (fire-and-forget) ─────────────────────
        asyncio.create_task(audit_log(user_input, answer, timings, blocked=False))

        return answer, timings


# ─────────────────────────────────────────────────────────────────────────────
# Latency Benchmark
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK_QUERIES = [
    "What is RAG and how does it improve LLM accuracy?",
    "Explain the difference between faithfulness and answer relevancy in RAGAS.",
    "How does BM25 differ from dense vector search?",
    "What is the purpose of reranking in a retrieval pipeline?",
    "Describe how Cohen's kappa measures inter-rater agreement.",
    "What are common failure modes in RAG systems?",
    "How does Llama Guard classify unsafe content?",
    "Explain the concept of position bias in LLM-as-Judge evaluation.",
    "What is PII and how can it be detected automatically?",
    "How does FAISS enable fast similarity search?",
    "What is the difference between context precision and context recall?",
    "How do input guardrails protect against prompt injection?",
    "What are SLOs and why are they important for AI systems?",
    "Explain the swap-and-average technique for pairwise evaluation.",
    "How does chunking strategy affect RAG performance?",
    "What is the purpose of a topic scope validator?",
    "How can latency be measured in a multi-layer pipeline?",
    "What is the difference between a self-hosted and API-based guardrail?",
    "Describe the 4-layer defense-in-depth architecture for LLM systems.",
    "What is semantic entropy in the context of hallucination detection?",
] * 5  # repeat to get 100 queries


async def run_benchmark(pipeline: GuardedPipeline, n: int = 100) -> pd.DataFrame:
    """Run latency benchmark on n queries."""
    print(f"\n[C.5] Running latency benchmark ({n} requests)...")
    queries = BENCHMARK_QUERIES[:n]
    all_timings = []

    for i, q in enumerate(queries):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] ...")
        try:
            _, timings = await pipeline.run(q)
            all_timings.append(timings)
        except Exception as e:
            print(f"  [{i+1}] Error: {e}")
            all_timings.append({"L1": 0, "L2": 0, "L3": 0})

    # Compute percentiles
    results = []
    print("\n[C.5] Latency Report:")
    print(f"{'Layer':<6} {'P50':>8} {'P95':>8} {'P99':>8} {'Target':>12} {'Status':>8}")
    print("-" * 55)

    targets = {"L1": 50, "L2": 99999, "L3": 100}  # ms
    for layer in ["L1", "L2", "L3"]:
        vals = [t.get(layer, 0) for t in all_timings if t.get(layer, 0) > 0]
        if not vals:
            continue
        p50 = np.percentile(vals, 50)
        p95 = np.percentile(vals, 95)
        p99 = np.percentile(vals, 99)
        target = targets.get(layer, 99999)
        status = "✅" if p95 < target else "⚠️"
        target_str = f"< {target}ms" if target < 99999 else "—"
        print(f"{layer:<6} {p50:>7.0f}ms {p95:>7.0f}ms {p99:>7.0f}ms {target_str:>12} {status:>8}")
        results.append({
            "layer": layer, "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1), "p99_ms": round(p99, 1),
            "target_ms": target, "passed": p95 < target,
        })

    df_results = pd.DataFrame(results)
    out_path = PHASE_C_DIR / "latency_benchmark.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\n[C.5] ✅ Benchmark saved → {out_path}")

    # Also save raw timings
    raw_df = pd.DataFrame(all_timings)
    raw_df.to_csv(PHASE_C_DIR / "latency_raw.csv", index=False)
    return df_results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    pipeline = GuardedPipeline()

    # Quick smoke test
    print("\n[C.5] Smoke test...")
    response, timings = await pipeline.run("What is RAG and how does it work?")
    print(f"  Response: {response[:100]}...")
    print(f"  Timings: { {k: f'{v:.0f}ms' for k, v in timings.items()} }")

    # Full benchmark
    await run_benchmark(pipeline, n=100)
    print("\n✅ Phase C.5 complete!")


if __name__ == "__main__":
    # Fix for module import
    sys.path.insert(0, str(Path(__file__).parent))
    # Rename module import to avoid conflict
    import importlib.util, types

    # Patch: load input_guard as phase_c.input_guard
    spec = importlib.util.spec_from_file_location(
        "phase_c.input_guard",
        Path(__file__).parent / "input_guard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_c.input_guard"] = mod
    spec.loader.exec_module(mod)

    spec2 = importlib.util.spec_from_file_location(
        "phase_c.output_guard",
        Path(__file__).parent / "output_guard.py"
    )
    mod2 = importlib.util.module_from_spec(spec2)
    sys.modules["phase_c.output_guard"] = mod2
    spec2.loader.exec_module(mod2)

    asyncio.run(main())
