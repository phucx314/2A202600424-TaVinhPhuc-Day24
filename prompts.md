# prompts.md — AI Prompt Log
# Lab 24: Full Evaluation & Guardrail System
# Academic Integrity: All AI-assisted prompts logged here per course policy

---

## Session 1 — 2026-05-12

### Prompt 1 — Architecture Planning
**Tool:** Claude Sonnet 4.6 (Antigravity assistant)
**Prompt:**
> "m xem cho t file pdf hướng dẫn làm lab ngày 24 này với... m hãy lên plan hoàn chỉnh để t có thể ăn trọn điểm lab này"

**What AI generated:** Full implementation plan covering Phase A–D with acceptance criteria, file structure, API cost estimates, and bonus recommendations.

**What I reviewed/modified:** Reviewed the plan, confirmed approach was correct, added Groq API registration step.

---

### Prompt 2 — RAG Adapter
**Tool:** Claude Sonnet 4.6
**Prompt:**
> "Continue" (after environment setup discussion — building lightweight FAISS-based RAG wrapper to avoid Qdrant dependency from Day 18)

**What AI generated:** `rag_adapter.py` — FAISS-based RAG with OpenAI embeddings, LLM generation, caching, async support.

**What I reviewed/modified:** Verified the async interface matches what `full_pipeline.py` expects. Confirmed index caching path is correct.

---

### Prompt 3 — Phase A Scripts
**Tool:** Claude Sonnet 4.6
**What AI generated:** `generate_testset.py`, `run_ragas.py`, `failure_analysis.py`, `scripts/run_eval.py`, `.github/workflows/eval-gate.yml`

**What I reviewed/modified:** Verified RAGAS import paths match version 0.4.3 API. Confirmed `testset_v1.csv` column names match grader expectations.

---

### Prompt 4 — Phase B Scripts
**Tool:** Claude Sonnet 4.6
**What AI generated:** `phase-b/judge_pipeline.py` — pairwise judge with swap-and-average, absolute scoring, kappa computation, bias analysis with chart.

**What I reviewed/modified:**
- Verified the winner-flipping logic in swap-and-average is correct (A↔B inversion)
- Confirmed bias chart uses matplotlib.use("Agg") for headless server compatibility
- Manually labeled `human_labels.csv` by reading `to_label.csv` answers

---

### Prompt 5 — Phase C Scripts
**Tool:** Claude Sonnet 4.6
**What AI generated:** `input_guard.py` (PII + topic + adversarial), `output_guard.py` (Llama Guard via Groq), `full_pipeline.py` (4-layer async stack)

**What I reviewed/modified:**
- Verified VN_PII regex patterns against sample Vietnamese PII data
- Tested `InputGuard.sanitize()` on edge cases (empty, very long)
- Confirmed `asyncio.gather()` pattern is correct for parallel L1 execution
- Reviewed unsafe test outputs to ensure they're within ethical testing norms (test-only, not deployed)

---

### Prompt 6 — Phase D Blueprint
**Tool:** Claude Sonnet 4.6
**What AI generated:** `phase-d/blueprint.md` — SLOs, Mermaid architecture diagram, 3 incident playbooks, cost analysis

**What I reviewed/modified:**
- Verified SLO thresholds match actual RAGAS benchmark targets from the lab spec
- Updated cost table with actual Groq pricing (free tier) after checking console.groq.com
- Added "Lessons Learned" section with real observations from running the code

---

## Notes
- All code was reviewed and tested before committing
- AI-generated code was treated as a starting point, not final output
- Where AI made errors (e.g., import paths for RAGAS 0.4.3 API changes), I identified and fixed them
- This log covers all AI interactions per academic integrity policy
