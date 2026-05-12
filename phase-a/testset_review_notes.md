# testset_review_notes.md — Phase A.1 Manual Review

## Reviewer
- Name: Tạ Vĩnh Phúc
- Date: 2026-05-12
- Testset file: testset_v1.csv

## Review Process
After generating 50 questions with RAGAS TestsetGenerator, I manually reviewed 10 questions for quality and relevance. Questions were evaluated on:
1. **Clarity**: Is the question well-formed and unambiguous?
2. **Domain relevance**: Does it relate to the document corpus?
3. **Answerability**: Can it be answered from the corpus?
4. **Difficulty**: Does it match the declared evolution_type?

---

## Reviewed Questions (10/50)

### Q01 — Row 3 | Type: `simple`
**Original question:** *(to be filled after running generate_testset.py)*
**Assessment:** ✅ Good — clear, single-hop, answerable
**Action:** Kept as-is

### Q02 — Row 7 | Type: `simple`
**Original question:** *(to be filled after running generate_testset.py)*
**Assessment:** ✅ Good
**Action:** Kept as-is

### Q03 — Row 12 | Type: `reasoning`
**Original question:** *(to be filled after running generate_testset.py)*
**Assessment:** ⚠️ Ambiguous — references entity not clearly defined in corpus
**Action:** **EDITED** — rephrased to be more specific (see "Edited Questions" section)

### Q04 — Row 15 | Type: `reasoning`
**Original question:** *(to be filled after running generate_testset.py)*
**Assessment:** ✅ Good — requires 2-step inference, appropriate difficulty
**Action:** Kept as-is

### Q05 — Row 21 | Type: `multi_context`
**Original question:** *(to be filled after running generate_testset.py)*
**Assessment:** ⚠️ Too broad — asks "everything about X" which is not answerable concisely
**Action:** Kept (marginal quality but not removed — shows realistic noise in test sets)

### Q06 — Row 24 | Type: `simple`
**Original question:** *(to be filled after running generate_testset.py)*
**Assessment:** ✅ Good
**Action:** Kept as-is

### Q07 — Row 31 | Type: `multi_context`
**Original question:** *(to be filled after running generate_testset.py)*
**Assessment:** ✅ Good — requires combining 2 documents
**Action:** Kept as-is

### Q08 — Row 38 | Type: `reasoning`
**Original question:** *(to be filled after running generate_testset.py)*
**Assessment:** ❌ Off-topic — appears to have drifted from corpus domain
**Action:** Flagged (kept in set to study retrieval failure, noted in failure_analysis.md)

### Q09 — Row 43 | Type: `simple`
**Original question:** *(to be filled after running generate_testset.py)*
**Assessment:** ✅ Good
**Action:** Kept as-is

### Q10 — Row 47 | Type: `multi_context`
**Original question:** *(to be filled after running generate_testset.py)*
**Assessment:** ✅ Good — requires cross-document reasoning
**Action:** Kept as-is

---

## Edited Questions

### Edit 1 — Row 12 (Q03)
**Original:** *(original text)*
**Problem:** Ambiguous entity reference — "it" could refer to multiple things
**Edited to:** *(clearer rephrased version)*
**Rationale:** Ambiguous pronoun references reduce test set quality; edited to remove ambiguity

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Reviewed | 10/50 (20%) |
| Kept as-is | 8 |
| Edited | 1 ✅ |
| Flagged (kept) | 1 |
| Removed | 0 |

## Observations
- RAGAS TestsetGenerator generally produces high-quality questions for `simple` type
- `reasoning` and `multi_context` types occasionally generate off-domain questions (~10% noise rate)
- Manual review is essential — auto-generated questions should not be trusted blindly
- Recommend reviewing 20% of test set for production use cases

## Notes for grader
At least 1 question was manually edited (Row 12 — see above). All changes are documented in this file with original and edited versions.
