# Lab 24 — Full Evaluation & Guardrail System

## Overview

This project implements a production-ready evaluation and guardrail system for a RAG (Retrieval-Augmented Generation) pipeline. The system answers three critical questions about AI in production:

1. **"Does this system work well?"** → Phase A: RAGAS automated evaluation across 4 metrics on a synthetic test set of 50 questions
2. **"When users attack it, does it hold up?"** → Phase C: Defense-in-depth 4-layer guardrail stack (PII redaction, topic validation, adversarial detection, Llama Guard output safety)
3. **"When it breaks, will we know in time?"** → Phase D: SLO definitions, architecture diagram, alert playbook, and cost analysis

## Setup

```bash
# Clone and enter
cd lab24-eval-guardrails-tavinhphuc

# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Configure environment
cp .env.example .env
# Edit .env: add OPENAI_API_KEY and GROQ_API_KEY
```

## Run Each Phase

```bash
# Phase A — RAGAS Evaluation
python phase-a/generate_testset.py      # generates testset_v1.csv
python phase-a/run_ragas.py             # evaluates 4 metrics
python phase-a/failure_analysis.py     # identifies failure clusters

# Phase B — LLM-as-Judge
python phase-b/judge_pipeline.py       # all B tasks in one script

# Phase C — Guardrails
python phase-c/input_guard.py          # PII + topic + adversarial tests
python phase-c/output_guard.py         # Llama Guard test
python phase-c/full_pipeline.py        # full stack + latency benchmark

# Verify YAML
python -c "import yaml; yaml.safe_load(open('.github/workflows/eval-gate.yml'))"
```

## Results Summary

### Phase A — RAGAS Evaluation
- **Test set:** 10 questions (evaluated with Python 3.11 + gpt-4o-mini)
| Metric | Target | Result | SLO | Priority |
| :--- | :--- | :--- | :--- | :--- |
| Faithfulness | ≥ 0.85 | 0.46 (corpus too small) | — | P2 |
| Answer Relevancy | ≥ 0.80 | 0.314 (corpus too small) | — | P2 |
| Context Precision | ≥ 0.70 | 0.559 (corpus too small) | — | P3 |
| Context Recall | ≥ 0.75 | 0.550 (corpus too small) | — | P3 |
| End-to-End P95 Latency | < 2.5s | 2.1s | 5 min | P1 |
| Guardrail Detection Rate | ≥ 90% | 90% | 1 hour | P2 |
| Guardrail False Positive Rate | < 10% | 0% | 1 hour | P2 |
- **Root cause:** Corpus from Day 18 contains only 3 FAISS chunks — insufficient coverage for diverse testset questions. Fix: expand corpus or increase chunk granularity.
- **Total eval cost:** ~$0.50 (gpt-4o-mini, 10 questions)
- **Failure clusters:** See [phase-a/failure_analysis.md](phase-a/failure_analysis.md)

### Phase B — LLM-as-Judge
- **Pairwise:** Swap-and-average on 30 questions, position bias mitigated
- **Cohen's kappa vs human:** 0.8182 (Almost Perfect) ✅
- **Biases quantified:** Position bias (mitigated) + length bias (identified)

### Phase C — Guardrails
- **PII detection rate:** 100% | Target ≥ 80% ✅
- **Topic validator accuracy:** 100% (after threshold tuning) | Target ≥ 75% ✅
- **Adversarial defense rate:** 90% | Target ≥ 70% ✅
- **Llama-3.1-8b detection:** 100% | Target ≥ 80% ✅
- **L1 P95 Latency:** 535ms | Target < 50ms ⚠️ (remote API overhead)
- **L3 P95 Latency:** 335ms | Target < 100ms ⚠️ (remote API overhead)

### Phase D — Blueprint
See full blueprint at [phase-d/blueprint.md](phase-d/blueprint.md)
- 7 SLOs defined with alert thresholds
- 4-layer architecture diagram (Mermaid)
- 3 incident playbooks
- Cost breakdown: ~$36/month for 100k queries

## Demo Video
*(YouTube link or local path — 5 minutes)*
1. RAGAS running live on 5 questions
2. LLM-Judge comparing 2 RAG versions
3. Adversarial attacks blocked by guardrails (DAN, jailbreak, PII)
4. Latency benchmark P50/P95/P99

## Lessons Learned
1. Swap-and-average is critical — without it, 60%+ of pairwise decisions are position-biased
2. Groq free tier is fully sufficient for Llama Guard — no GPU needed for lab scale
3. Async parallel execution in L1 cuts guardrail overhead from ~150ms to ~50ms
4. Manual review of generated test set is non-negotiable — ~10% of RAGAS-generated questions drift off-domain

## Repository Structure
```
├── phase-a/          # RAGAS evaluation scripts + outputs
├── phase-b/          # LLM-as-Judge pipeline + outputs
├── phase-c/          # Guardrail stack + benchmark
├── phase-d/          # Blueprint document
├── .github/          # CI/CD eval gate workflow
├── scripts/          # Shared scripts (run_eval.py)
├── rag_adapter.py    # Lightweight FAISS-based RAG (Day 18 corpus)
├── prompts.md        # AI prompt log (academic integrity)
└── requirements.txt  # Pinned dependencies
```

## Author
Tạ Vĩnh Phúc — AICB Program, VinUniversity 2026
