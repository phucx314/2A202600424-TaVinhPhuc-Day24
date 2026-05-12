# testset_review_notes.md — Phase A.1 Manual Review

## Reviewer
- Name: Tạ Vĩnh Phúc
- Date: 2026-05-12
- Testset file: testset_v1.csv (50 questions)

## Review Process
After generating questions with RAGAS TestsetGenerator from the Day 18 corpus (HR policy + data protection documents), I manually reviewed 10 questions for quality and relevance. Questions were evaluated on:
1. **Clarity**: Is the question well-formed and unambiguous?
2. **Domain relevance**: Does it relate to the document corpus?
3. **Answerability**: Can it be answered from the corpus?
4. **Difficulty**: Does it match the declared evolution_type?

---

## Reviewed Questions (10/50)

### Q01 — Type: `simple`
**Question:** Thời gian thử việc đối với công việc văn phòng là bao lâu?
**Assessment:** ✅ Clear, relevant, directly answerable from HR policy doc.
**Action:** Kept as-is.

### Q02 — Type: `simple`
**Question:** Lao động nữ được nghỉ thai sản trong bao lâu?
**Assessment:** ✅ Clear and specific. Ground truth matches document.
**Action:** Kept as-is.

### Q03 — Type: `simple`
**Question:** Mật khẩu phải có ít nhất bao nhiêu ký tự và bao gồm những yếu tố nào?
**Assessment:** ✅ Good factual question, well-formed.
**Action:** Kept as-is.

### Q04 — Type: `reasoning`
**Question:** If an employee in an office job starts on January 1st and successfully completes their probation period, when can they expect to sign their official employment contract?
**Assessment:** ✅ Multi-step reasoning required (60-day probation → March 2). Good difficulty.
**Action:** Kept as-is.

### Q05 — Type: `reasoning`
**Question:** If an employee has worked for 12 years and wants to take a vacation, how many total paid leave days do they have?
**Assessment:** ⚠️ Original auto-generated version was slightly ambiguous — did not clarify if the 1-day/5-years bonus is cumulative.
**Action:** ✏️ **EDITED** ground truth to clarify: "12 base days + 2 bonus days (for 10 years of service) = 14 total days."

### Q06 — Type: `simple`
**Question:** Nhân viên phải làm gì khi kết thúc hợp đồng với công ty?
**Assessment:** ✅ Directly answerable, relevant to offboarding policy.
**Action:** Kept as-is.

### Q07 — Type: `multi_context`
**Question:** How can RAG systems ensure the ethical use of personal data in accordance with Vietnam's Nghị định 13/2023/NĐ-CP?
**Assessment:** ⚠️ This question requires combining RAG architecture knowledge + legal knowledge. The auto-generated ground truth was too generic.
**Action:** Kept but flagged — likely to score low on context_recall since corpus coverage is limited.

### Q08 — Type: `simple`
**Question:** What is the maximum training cost support per employee per year?
**Assessment:** ✅ Clear factual question, well-grounded in the document.
**Action:** Kept as-is.

### Q09 — Type: `reasoning`
**Question:** What commitment must employees make after receiving training support valued over 10 million VND?
**Assessment:** ✅ Requires inference from training policy. Good reasoning question.
**Action:** Kept as-is.

### Q10 — Type: `multi_context`
**Question:** How does the evaluation method of LLM-as-Judge mitigate biases, and how might this relate to the approval process for employee leave requests?
**Assessment:** ❌ Cross-domain question mixing LLM evaluation concepts with HR policy — irrelevant connection. This is a hallucination artifact from the generator.
**Action:** Flagged as low-quality. Kept in set to observe how RAGAS scores it (expected: low answer_relevancy).

---

## Summary of Findings
- **8/10** questions reviewed were acceptable quality
- **1 question edited** (Q05): Clarified ground truth for multi-year leave calculation
- **1 question flagged** (Q10): Cross-domain mismatch, kept to observe failure mode
- Distribution verified: `simple=50%, reasoning=25%, multi_context=25%` ✅

## Recommendation
For production use, recommend manual review of 100% of questions and removal of cross-domain artifacts before running expensive RAGAS evaluation.
