"""
analysis/robustness_bootstrap.py — paired bootstrap CIs on specificity
differences (structured - holistic) for each OSS model × variant cell.

Determines whether the two reversals in the robustness check are
statistically significant or within noise.

Run from repo root:
  python analysis/robustness_bootstrap.py
"""
import json, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from metrics import paired_bootstrap_comparison

JSONL = Path(__file__).parent / "robustness_dev.jsonl"


def load_groups():
    groups = defaultdict(list)
    with JSONL.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("pred_label") is not None and not r.get("api_error", False):
                groups[(r["model"], r["style"], r["variant"])].append(r)
    return groups


def main():
    groups = load_groups()
    oss_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    variants   = ["v0", "v1", "v2"]

    print("=" * 80)
    print("ROBUSTNESS BOOTSTRAP — specificity CIs for OSS models")
    print("Testing: is structured specificity significantly different from holistic?")
    print("=" * 80)

    for model in oss_models:
        print(f"\n  MODEL: {model}")
        print(f"  {'variant':8s} {'S_spec':>7s} {'H_spec':>7s} {'diff':>7s} "
              f"{'CI_lo':>7s} {'CI_hi':>7s} {'verdict'}")
        print(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*40}")
        for v in variants:
            s_recs = groups.get((model, "structured", v), [])
            h_recs = groups.get((model, "holistic",   v), [])
            if not s_recs or not h_recs:
                print(f"  {v:8s}  MISSING")
                continue
            r = paired_bootstrap_comparison(s_recs, h_recs, metric="specificity",
                                            n_boot=5000, seed=42, silent=True)
            s_spec = r["structured"]
            h_spec = r["holistic"]
            diff   = r["observed_diff"]
            lo, hi = r["ci_lo"], r["ci_hi"]
            if lo > 0:
                verdict = "STRUCTURED > HOLISTIC *"
            elif hi < 0:
                verdict = "HOLISTIC > STRUCTURED *"
            else:
                verdict = "null (noise)"
            print(f"  {v:8s} {s_spec:7.3f} {h_spec:7.3f} {diff:+7.3f} "
                  f"{lo:+7.3f} {hi:+7.3f}  {verdict}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("  'null (noise)' = the reversal or advantage is within sampling variance.")
    print("  Report only CI-separated differences as claims.")
    print("=" * 80)


if __name__ == "__main__":
    main()