# Failure Cluster Analysis — Phase A.3

## Bottom 10 Questions (Lowest Average RAGAS Score)

| # | Question (truncated) | Type | FA | AN | CO | CO | Avg | Cluster |
|---|---|---|---|---|---|---|---|---|
| 1 | Dữ liệu cá nhân cơ bản bao gồm những thông ti... | ? | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | C1 |
| 2 | What are the three types of input guardrails ... | ? | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | C1 |
| 3 | If a RAG pipeline evaluates with a faithfulne... | ? | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | C1 |
| 4 | What are the target thresholds for faithfulne... | ? | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| 5 | What measures are taken to protect sensitive ... | ? | 0.00 | 0.00 | 1.00 | 0.50 | 0.37 | — |
| 6 | Lao động nữ được nghỉ thai sản trong bao lâu?... | ? | 1.00 | 0.28 | 1.00 | 1.00 | 0.82 | — |
| 7 | Khi nhân viên làm việc từ xa, công ty cần lưu... | ? | 0.80 | 0.51 | 1.00 | 1.00 | 0.83 | — |
| 8 | If an employee has worked for 12 years and wa... | ? | 0.80 | 0.79 | 0.89 | 1.00 | 0.87 | — |
| 9 | Mật khẩu phải có ít nhất bao nhiêu ký tự và b... | ? | 1.00 | 0.55 | 1.00 | 1.00 | 0.89 | — |
| 10 | What is the maximum training cost support per... | ? | 1.00 | 1.00 | 0.70 | 1.00 | 0.92 | — |

*Metrics: F=Faithfulness, AR=Answer Relevancy, CP=Context Precision, CR=Context Recall*

---

## Clusters Identified


### Cluster C1: Hallucination / Low Faithfulness

**Pattern:** Questions where the model generates answers not grounded in retrieved context.

**Examples:**
  - "Dữ liệu cá nhân cơ bản bao gồm những thông tin nào theo Nghị định 13/2023/NĐ-CP?..."
  - "What are the three types of input guardrails mentioned for LLM systems?..."
  - "If a RAG pipeline evaluates with a faithfulness score of 0.90, an answer relevan..."

**Root cause:** Retriever returns tangentially related chunks; LLM then confabulates details.

**Proposed fix:**
  1. Increase `top_k` from 5→8 to retrieve more context.
  2. Add faithfulness check as pre-filter: if LLM claims fact not in context, substitute 'I don't know'.
  3. Use stricter system prompt: 'Only state facts explicitly mentioned in the context.'


---

## Summary

| Cluster | Count | Primary Issue | Recommended Fix Priority |
|---------|-------|---------------|--------------------------|
| C1 | 3 | Hallucination / Low Faithfulness | High |