"""
Phase A — Task A.4: CI/CD Eval Gate Script
Run with: python scripts/run_eval.py --threshold faithfulness=0.85
Exit code 1 if any metric below threshold → blocks PR merge.
"""

import sys
import json
import argparse
from pathlib import Path


def parse_thresholds(args: list[str]) -> dict[str, float]:
    """Parse --threshold key=value pairs."""
    thresholds = {}
    for item in args:
        try:
            key, val = item.split("=")
            thresholds[key.strip()] = float(val.strip())
        except ValueError:
            print(f"[eval-gate] WARNING: Could not parse threshold '{item}', skipping.")
    return thresholds


def load_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        print(f"[eval-gate] ERROR: Summary not found at {summary_path}")
        sys.exit(1)
    with open(summary_path) as f:
        return json.load(f)


def check_thresholds(summary: dict, thresholds: dict[str, float]) -> bool:
    """Returns True if all metrics pass, False if any fail."""
    all_pass = True
    print("\n[eval-gate] Checking RAGAS thresholds...")
    print(f"{'Metric':<30} {'Score':>8} {'Threshold':>12} {'Status':>8}")
    print("-" * 62)

    for metric, threshold in thresholds.items():
        score = summary.get(metric)
        if score is None:
            print(f"{metric:<30} {'N/A':>8} {threshold:>12.3f} {'⚠️ MISSING':>8}")
            all_pass = False
            continue
        passed = score >= threshold
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{metric:<30} {score:>8.4f} {threshold:>12.3f} {status:>8}")
        if not passed:
            all_pass = False

    print("-" * 62)
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="RAGAS Eval Gate for CI/CD")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Threshold in format metric=value (can repeat)",
    )
    parser.add_argument(
        "--summary",
        default="phase-a/ragas_summary.json",
        help="Path to ragas_summary.json",
    )
    args = parser.parse_args()

    # Default thresholds if none provided
    thresholds = parse_thresholds(args.threshold) if args.threshold else {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.70,
        "context_recall": 0.75,
    }

    summary_path = Path(args.summary)
    summary = load_summary(summary_path)
    all_pass = check_thresholds(summary, thresholds)

    if all_pass:
        print("\n✅ All metrics pass. PR can merge.")
        sys.exit(0)
    else:
        print("\n❌ Some metrics below threshold. Blocking PR merge.")
        sys.exit(1)


if __name__ == "__main__":
    main()
