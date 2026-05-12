# prompts.md — AI Prompt Log
# Lab 24: Full Evaluation & Guardrail System
# Academic Integrity: All AI-assisted prompts logged here per course policy

---

## Session 1 — 2026-05-12

### Prompt 1 — System Architecture & Planning
**Tool:** Claude Sonnet 4.6
**Prompt:**
> "Please analyze the Lab 24 PDF instructions and help me outline a technical implementation plan that covers all requirements from Phase A to Phase D. I need to ensure the architecture is production-ready and includes RAGAS evaluation, LLM-as-a-Judge, and a multi-layer guardrail system."

**What AI generated:** A structured implementation roadmap including file hierarchy, dependency selection (RAGAS, Groq, Presidio), and a verification plan.

**What I reviewed/modified:** I verified the proposed architecture against the lab requirements, adjusted the PII detection strategy to include Vietnamese-specific regex, and set up the local environment.

---

### Prompt 2 — RAG Adapter Implementation
**Tool:** Claude Sonnet 4.6
**Prompt:**
> "Help me implement a lightweight RAG adapter using FAISS and OpenAI embeddings. The system should support asynchronous queries and local indexing to optimize performance for the evaluation pipeline."

**What AI generated:** `rag_adapter.py` providing a clean interface for retrieval and generation with FAISS vector store.

**What I reviewed/modified:** Optimized the chunking logic and added error handling for API timeouts.

---

### Prompt 3 — Guardrails & Latency Benchmarking
**Tool:** Claude Sonnet 4.6
**Prompt:**
> "I need to implement a 4-layer guardrail stack as described in Phase C. Please help me structure the `InputGuard` and `OutputGuard` classes using asyncio to run checks in parallel and minimize latency overhead."

**What AI generated:** Asynchronous pipeline structure for parallel execution of PII and Topic guards.

**What I reviewed/modified:** Adjusted the similarity threshold for the Topic Guard based on empirical testing with the Day 18 corpus.

---

### Prompt 4 — Evaluation & Report Generation
**Tool:** Claude Sonnet 4.6
**Prompt:**
> "Assist me in calculating Cohen's Kappa score for the LLM judge calibration and generating a summary report for the RAGAS metrics."

**What AI generated:** Statistical calculation scripts for Kappa and markdown templates for the bias analysis report.

**What I reviewed/modified:** Performed the manual human labeling for the 10-pair sample and verified the agreement score.
