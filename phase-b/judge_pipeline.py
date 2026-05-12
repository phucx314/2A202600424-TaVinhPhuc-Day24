"""
Phase B — Task B.1 + B.2 + B.3 + B.4: LLM-as-Judge Pipeline
Pairwise comparison (swap-and-average), absolute scoring, Cohen's kappa, bias report.

Run all: python phase-b/judge_pipeline.py
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
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

PHASE_B_DIR = Path(__file__).parent
TESTSET_PATH = Path(__file__).parent.parent / "phase-a" / "testset_v1.csv"
PAIRWISE_OUT = PHASE_B_DIR / "pairwise_results.csv"
ABSOLUTE_OUT = PHASE_B_DIR / "absolute_scores.csv"
HUMAN_LABELS_OUT = PHASE_B_DIR / "human_labels.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────
PAIRWISE_PROMPT = PromptTemplate.from_template("""You are an impartial evaluator. Compare two answers to the same question.

Question: {question}

Answer A: {answer_a}

Answer B: {answer_b}

Rate based on:
- Factual accuracy
- Relevance to question
- Conciseness and clarity

Output JSON only, no markdown fences:
{{"winner": "A" or "B" or "tie", "reason": "one sentence explanation"}}""")

ABSOLUTE_PROMPT = PromptTemplate.from_template("""Score the following answer on 4 dimensions, each 1-5 scale:
1. accuracy (1=many errors, 5=fully accurate)
2. relevance (1=off-topic, 5=directly answers the question)
3. conciseness (1=verbose/rambling, 5=appropriately brief)
4. helpfulness (1=unclear/unhelpful, 5=actionable and clear)

Question: {question}
Answer: {answer}

Output JSON only, no markdown fences:
{{"accuracy": int, "relevance": int, "conciseness": int, "helpfulness": int}}""")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def parse_json_robust(text: str, fallback: dict) -> dict:
    """Parse JSON from LLM output with multiple fallback strategies."""
    if not text:
        return fallback
    # Strip markdown fences
    text = text.strip()
    for fence in ["```json", "```JSON", "```"]:
        text = text.replace(fence, "")
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting first {...} block
    import re
    match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return fallback


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ─────────────────────────────────────────────────────────────────────────────
# Task B.1 — Pairwise Judge with Swap-and-Average
# ─────────────────────────────────────────────────────────────────────────────
def pairwise_judge_with_swap(
    question: str, ans_v1: str, ans_v2: str, llm: ChatOpenAI
) -> tuple[str, str, str]:
    """
    Run pairwise comparison TWICE with swapped order to mitigate position bias.
    Returns: (run1_winner, run2_winner, final_winner_after_swap)
    """
    # Run 1: v1=A, v2=B
    prompt1 = PAIRWISE_PROMPT.format(question=question, answer_a=ans_v1, answer_b=ans_v2)
    out1 = llm.invoke(prompt1)
    r1 = parse_json_robust(out1.content, {"winner": "tie", "reason": "parse error"})
    run1_winner = r1.get("winner", "tie")

    # Run 2: v2=A, v1=B  (swapped)
    prompt2 = PAIRWISE_PROMPT.format(question=question, answer_a=ans_v2, answer_b=ans_v1)
    out2 = llm.invoke(prompt2)
    r2 = parse_json_robust(out2.content, {"winner": "tie", "reason": "parse error"})
    raw_run2 = r2.get("winner", "tie")

    # Flip run2 winner back to original perspective (v1=A, v2=B)
    if raw_run2 == "A":
        run2_winner = "B"   # swapped A was actually v2
    elif raw_run2 == "B":
        run2_winner = "A"   # swapped B was actually v1
    else:
        run2_winner = "tie"

    # Final: both agree → winner; disagree → tie
    final = run1_winner if run1_winner == run2_winner else "tie"
    return run1_winner, run2_winner, final


def run_pairwise_evaluation(questions: list[str], n: int = 30) -> pd.DataFrame:
    """Run pairwise evaluation on first n questions."""
    from rag_adapter import RAGPipeline, RAGPipelineV2

    print(f"\n[B.1] Running pairwise evaluation on {n} questions...")
    rag_v1 = RAGPipeline()
    rag_v2 = RAGPipelineV2()
    llm = get_llm()

    records = []
    questions = questions[:n]
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{n}] {q[:55]}...")
        try:
            ans_v1, _ = rag_v1.query(q)
            ans_v2, _ = rag_v2.query(q)
            run1, run2, final = pairwise_judge_with_swap(q, ans_v1, ans_v2, llm)
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
            ans_v1, ans_v2 = "Error", "Error"
            run1, run2, final = "tie", "tie", "tie"

        records.append({
            "question": q,
            "answer_v1": ans_v1,
            "answer_v2": ans_v2,
            "run1_winner": run1,
            "run2_winner": run2,
            "winner_after_swap": final,
            "len_v1": len(ans_v1),
            "len_v2": len(ans_v2),
        })

    df = pd.DataFrame(records)
    df.to_csv(PAIRWISE_OUT, index=False)
    print(f"[B.1] ✅ Saved → {PAIRWISE_OUT}")

    # Summary
    vc = df["winner_after_swap"].value_counts()
    print(f"\n  Winner distribution: {vc.to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Task B.2 — Absolute Scoring
# ─────────────────────────────────────────────────────────────────────────────
def run_absolute_scoring(df_pairwise: pd.DataFrame) -> pd.DataFrame:
    """Score v1 answers absolutely on 4 dimensions."""
    print(f"\n[B.2] Running absolute scoring on {len(df_pairwise)} questions...")
    llm = get_llm()
    records = []

    for i, row in df_pairwise.iterrows():
        print(f"  [{i+1}/{len(df_pairwise)}] scoring...")
        try:
            prompt = ABSOLUTE_PROMPT.format(
                question=row["question"], answer=row["answer_v1"]
            )
            out = llm.invoke(prompt)
            scores = parse_json_robust(out.content, {
                "accuracy": 3, "relevance": 3, "conciseness": 3, "helpfulness": 3
            })
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
            scores = {"accuracy": 3, "relevance": 3, "conciseness": 3, "helpfulness": 3}

        # Clamp to 1-5
        dims = ["accuracy", "relevance", "conciseness", "helpfulness"]
        for d in dims:
            scores[d] = max(1, min(5, int(scores.get(d, 3))))

        scores["overall"] = round(sum(scores[d] for d in dims) / 4, 2)
        scores["question"] = row["question"]
        records.append(scores)

    df_abs = pd.DataFrame(records)
    df_abs.to_csv(ABSOLUTE_OUT, index=False)
    print(f"[B.2] ✅ Saved → {ABSOLUTE_OUT}")
    print(f"  Mean overall score: {df_abs['overall'].mean():.2f}/5.0")
    return df_abs


# ─────────────────────────────────────────────────────────────────────────────
# Task B.3 — Human Labels Template + Cohen's Kappa
# ─────────────────────────────────────────────────────────────────────────────
def generate_human_labels_template(df_pairwise: pd.DataFrame) -> pd.DataFrame:
    """Generate template for human labeling (first 10 rows)."""
    sample = df_pairwise.sample(min(10, len(df_pairwise)), random_state=42).reset_index(drop=True)

    # Save labeling sheet
    label_sheet = sample[["question", "answer_v1", "answer_v2"]].copy()
    label_sheet.index.name = "question_id"
    label_sheet.to_csv(PHASE_B_DIR / "to_label.csv")
    print(f"\n[B.3] Labeling sheet saved → {PHASE_B_DIR / 'to_label.csv'}")

    # Pre-filled human labels (you MUST review these manually!)
    # These are placeholder values — open to_label.csv and judge each pair yourself
    human_records = []
    for i, row in sample.iterrows():
        # Auto-judge based on length heuristic as placeholder
        # ⚠️  REPLACE with your actual human judgment!
        len_v1 = len(row.get("answer_v1", ""))
        len_v2 = len(row.get("answer_v2", ""))
        auto_winner = "A" if len_v1 > len_v2 else ("B" if len_v2 > len_v1 else "tie")
        human_records.append({
            "question_id": i,
            "question_preview": row["question"][:60],
            "human_winner": auto_winner,  # ← CHANGE THIS after reading both answers
            "confidence": "medium",       # ← low / medium / high
            "notes": "Auto-placeholder — replace with your judgment",
        })

    df_human = pd.DataFrame(human_records)
    df_human.to_csv(HUMAN_LABELS_OUT, index=False)
    print(f"[B.3] ✅ Human labels template → {HUMAN_LABELS_OUT}")
    print("  ⚠️  IMPORTANT: Open human_labels.csv and manually judge each pair!")
    print("     Compare answer_v1 vs answer_v2 in to_label.csv, then update human_winner column.")
    return df_human, sample


def compute_cohen_kappa(df_human: pd.DataFrame, df_pairwise: pd.DataFrame) -> float:
    """Compute Cohen's kappa between human and judge labels."""
    from sklearn.metrics import cohen_kappa_score

    # Align on question_id
    human_labels = df_human.set_index("question_id")["human_winner"].tolist()
    sample_indices = df_human["question_id"].tolist()
    judge_labels = df_pairwise.iloc[sample_indices]["winner_after_swap"].tolist()

    # Normalize labels
    normalize = lambda x: x.upper().strip() if isinstance(x, str) else "TIE"
    human_norm = [normalize(l) for l in human_labels]
    judge_norm = [normalize(l) for l in judge_labels]

    try:
        kappa = cohen_kappa_score(human_norm, judge_norm)
    except Exception as e:
        print(f"[B.3] ⚠️  Kappa computation error: {e}")
        kappa = 0.0

    # Interpretation
    if kappa < 0:
        interp = "WORSE than chance — judge systematically disagrees with human"
    elif kappa < 0.2:
        interp = "Slight agreement — judge not reliable"
    elif kappa < 0.4:
        interp = "Fair agreement — weak, needs improvement"
    elif kappa < 0.6:
        interp = "Moderate agreement — usable for monitoring but not production decisions"
    elif kappa < 0.8:
        interp = "Substantial agreement — production-ready ✅"
    else:
        interp = "Almost perfect agreement — excellent calibration"

    print(f"\n[B.3] Cohen's Kappa: {kappa:.3f}")
    print(f"      Interpretation: {interp}")

    kappa_result = {
        "kappa": round(kappa, 4),
        "interpretation": interp,
        "n_pairs": len(human_norm),
        "human_labels": human_norm,
        "judge_labels": judge_norm,
    }
    with open(PHASE_B_DIR / "kappa_result.json", "w") as f:
        json.dump(kappa_result, f, indent=2)

    return kappa


# ─────────────────────────────────────────────────────────────────────────────
# Task B.4 — Bias Report
# ─────────────────────────────────────────────────────────────────────────────
def generate_bias_report(df: pd.DataFrame, kappa: float) -> None:
    """Quantify position bias and length bias, generate chart + markdown report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n[B.4] Generating bias analysis...")

    # ── Bias 1: Position bias ────────────────────────────────
    total = len(df)
    run1_a_wins = (df["run1_winner"] == "A").sum()
    run1_b_wins = (df["run1_winner"] == "B").sum()
    position_bias_pct = run1_a_wins / total * 100
    position_bias_flag = position_bias_pct > 55

    # ── Bias 2: Length bias ──────────────────────────────────
    df["len_diff"] = df["len_v2"] - df["len_v1"]
    b_wins_when_longer = ((df["winner_after_swap"] == "B") & (df["len_diff"] > 0)).sum()
    b_total_longer = (df["len_diff"] > 0).sum()
    a_wins_when_longer_v1 = ((df["winner_after_swap"] == "A") & (df["len_v1"] > df["len_v2"])).sum()
    a_total_longer_v1 = (df["len_v1"] > df["len_v2"]).sum()

    length_bias_pct_b = (b_wins_when_longer / b_total_longer * 100) if b_total_longer > 0 else 0
    length_bias_flag = length_bias_pct_b > 60

    # ── Chart ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LLM Judge Bias Analysis", fontsize=14, fontweight="bold")

    # Position bias bar chart
    ax1 = axes[0]
    categories = ["A wins (first)", "B wins (first)", "Tie"]
    values = [
        (df["run1_winner"] == "A").sum(),
        (df["run1_winner"] == "B").sum(),
        (df["run1_winner"] == "tie").sum(),
    ]
    colors = ["#3498db", "#e74c3c", "#95a5a6"]
    bars = ax1.bar(categories, values, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(y=total / 2, color="black", linestyle="--", alpha=0.5, label="Expected (no bias)")
    ax1.set_title(f"Position Bias\n(A-first win rate: {position_bias_pct:.1f}%)")
    ax1.set_ylabel("Count")
    ax1.legend()
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(val), ha="center", va="bottom")

    # Length bias scatter
    ax2 = axes[1]
    winner_color = df["winner_after_swap"].map({"A": "#3498db", "B": "#e74c3c", "tie": "#95a5a6"})
    ax2.scatter(df["len_v1"], df["len_v2"], c=winner_color, alpha=0.6, edgecolors="white", linewidth=0.5)
    ax2.plot([0, max(df["len_v1"].max(), df["len_v2"].max())],
             [0, max(df["len_v1"].max(), df["len_v2"].max())],
             "k--", alpha=0.4, label="Equal length")
    ax2.set_xlabel("Length of Answer V1")
    ax2.set_ylabel("Length of Answer V2")
    ax2.set_title(f"Length Bias\n(B wins when longer: {length_bias_pct_b:.1f}%)")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color="#3498db", label="A wins"),
        Patch(color="#e74c3c", label="B wins"),
        Patch(color="#95a5a6", label="Tie"),
    ]
    ax2.legend(handles=legend_elements)

    plt.tight_layout()
    chart_path = PHASE_B_DIR / "bias_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[B.4] Chart saved → {chart_path}")

    # ── Markdown report ──────────────────────────────────────
    report = f"""# Judge Bias Analysis Report — Phase B.4

## Summary

| Bias Type | Measured | Threshold | Flagged? |
|-----------|----------|-----------|----------|
| Position bias (A-first win rate) | {position_bias_pct:.1f}% | > 55% | {'⚠️ YES' if position_bias_flag else '✅ No'} |
| Length bias (B wins when longer) | {length_bias_pct_b:.1f}% | > 60% | {'⚠️ YES' if length_bias_flag else '✅ No'} |

## Bias 1: Position Bias

**Method:** Measured how often Answer A wins when listed first in Run 1 (before swap).

**Results:**
- A wins as first: {run1_a_wins}/{total} = **{position_bias_pct:.1f}%**
- B wins as first: {run1_b_wins}/{total} = {run1_b_wins/total*100:.1f}%
- Expected (no bias): 50%

**Interpretation:** {'⚠️ Position bias detected — judge prefers first answer. Swap-and-average mitigates this.' if position_bias_flag else '✅ No significant position bias detected. Judge is consistent regardless of answer order.'}

**Mitigation applied:** Swap-and-average — each pair evaluated twice with flipped order, final winner requires both runs to agree.

---

## Bias 2: Length Bias

**Method:** Correlation between answer length difference and judge preference.

**Results:**
- When Answer B is longer: B wins {b_wins_when_longer}/{b_total_longer} = **{length_bias_pct_b:.1f}%**
- When Answer A is longer: A wins {a_wins_when_longer_v1}/{a_total_longer_v1} = {a_wins_when_longer_v1/a_total_longer_v1*100:.1f}% (if applicable)
- Avg length V1: {df['len_v1'].mean():.0f} chars | Avg length V2: {df['len_v2'].mean():.0f} chars

**Interpretation:** {'⚠️ Length bias detected — judge tends to prefer longer answers. Longer ≠ Better.' if length_bias_flag else '✅ No significant length bias. Judge evaluates quality, not quantity.'}

**Mitigation strategy:** Normalize answer length in prompt ("Prefer concise answers. Length alone is not a quality signal.") or use absolute scoring rubric which explicitly scores conciseness.

---

## Human Calibration (Cohen's Kappa)

**Kappa score:** {kappa:.4f}

**Interpretation:** {'Substantial agreement — judge is production-ready ✅' if kappa >= 0.6 else 'Moderate/fair agreement — monitor judge decisions before full deployment'}

---

## Chart

![Bias Analysis](bias_chart.png)

---

## Conclusion & Mitigation Strategy

1. **Swap-and-average** (already implemented in B.1) — mitigates position bias effectively
2. **Absolute scoring rubric** (B.2) — penalizes verbosity via conciseness dimension  
3. **Future improvement:** Add confidence scoring — only act on judge decisions with high confidence
4. **Monitor kappa** — re-calibrate every 500 evaluations or when domain shifts
"""

    report_path = PHASE_B_DIR / "judge_bias_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[B.4] ✅ Bias report → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load questions from testset
    if not TESTSET_PATH.exists():
        print(f"ERROR: {TESTSET_PATH} not found. Run phase-a/generate_testset.py first.")
        sys.exit(1)

    df_testset = pd.read_csv(TESTSET_PATH)
    questions = df_testset["question"].tolist()

    # B.1 — Pairwise
    df_pairwise = run_pairwise_evaluation(questions, n=30)

    # B.2 — Absolute scoring
    df_absolute = run_absolute_scoring(df_pairwise)

    # B.3 — Human labels + kappa
    df_human, sample = generate_human_labels_template(df_pairwise)
    kappa = compute_cohen_kappa(df_human, df_pairwise)

    # B.4 — Bias report + chart
    generate_bias_report(df_pairwise, kappa)

    print("\n✅ Phase B complete!")
    print(f"  Pairwise: {PAIRWISE_OUT}")
    print(f"  Absolute: {ABSOLUTE_OUT}")
    print(f"  Kappa: {kappa:.3f}")
