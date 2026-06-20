"""
analysis/phase2_checks.py — five pre-publication checks.

Run from repo root:
  python analysis/phase2_checks.py --results results --split dev

Covers:
  1. Bootstrap CIs on the four key GPT-4o-mini gaps (no new model calls).
  2. Claims-only sanity check: verify the claim list is identical for
     faithful and omission members of each pair, and inspect what the
     monitor actually sees.
  3. Power analysis for the test split.
  4. Seed stability check (re-run with seed=99, different few-shot order).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "eval"))

from schema import load_jsonl, iter_split, example_pool
from metrics import paired_bootstrap_comparison, bootstrap_ci, compute_metrics


# ── helpers ────────────────────────────────────────────────────────────────────

def load_condition(results_dir: Path, split: str,
                   style: str, level: str, model_tag: str) -> list[dict] | None:
    """Load result dicts for one (style, level, model) condition."""
    for f in sorted(results_dir.glob(f"{split}_{style}_{level}_{model_tag}*.json")):
        data = json.loads(f.read_text())
        if "results" in data:
            return data["results"]
    return None


def load_all_conditions(results_dir: Path, split: str) -> dict:
    """
    Returns nested dict: conditions[model][style][level] = list[dict]
    model_tag patterns: llama3_latest, gpt_4o_mini, gpt_4o
    """
    model_tags = {
        "llama3:latest": "llama3_latest",
        "gpt-4o-mini":   "gpt_4o_mini",
        "gpt-4o":        "gpt_4o",
    }
    styles = ["holistic", "structured"]
    levels = ["full", "claims_only", "output_only"]
    out = {}
    for model, tag in model_tags.items():
        out[model] = {}
        for style in styles:
            out[model][style] = {}
            for level in levels:
                r = load_condition(results_dir, split, style, level, tag)
                if r:
                    out[model][style][level] = r
    return out


# ── 1. Bootstrap CIs on key gaps ──────────────────────────────────────────────

def check_1_bootstrap_gaps(conds: dict) -> None:
    print("\n" + "=" * 68)
    print("CHECK 1: Paired bootstrap CIs on key GPT-4o-mini gaps")
    print("=" * 68)

    model = "gpt-4o-mini"
    mc = conds.get(model, {})
    if not mc:
        print("  No GPT-4o-mini results found.")
        return

    comparisons = [
        ("H1 at full: structured vs holistic",
         mc.get("structured", {}).get("full"),
         mc.get("holistic",   {}).get("full")),
        ("Source-access (holistic): full vs output_only",
         mc.get("holistic", {}).get("full"),
         mc.get("holistic", {}).get("output_only")),
        ("Source-access (structured): full vs output_only  [NOVEL CLAIM]",
         mc.get("structured", {}).get("full"),
         mc.get("structured", {}).get("output_only")),
        ("Output-only gap: structured vs holistic  [MOST NOVEL]",
         mc.get("structured", {}).get("output_only"),
         mc.get("holistic",   {}).get("output_only")),
        ("Claims-only vs full (holistic)",
         mc.get("holistic", {}).get("claims_only"),
         mc.get("holistic", {}).get("full")),
        ("Claims-only vs full (structured)",
         mc.get("structured", {}).get("claims_only"),
         mc.get("structured", {}).get("full")),
    ]

    for label, ra, rb in comparisons:
        if not ra or not rb:
            print(f"\n  [{label}]  MISSING DATA")
            continue
        print(f"\n  [{label}]")
        comp = paired_bootstrap_comparison(ra, rb, seed=42, n_boot=5000)
        obs_a = comp["ba_structured"]
        obs_b = comp["ba_holistic"]
        print(f"    A={obs_a:.4f}  B={obs_b:.4f}  diff={comp['observed_diff']:+.4f}")
        print(f"    95% CI: [{comp['ci_lo']:+.4f}, {comp['ci_hi']:+.4f}]  "
              f"p={comp['p_value']:.4f}")
        print(f"    {comp['verdict']}")


# ── 2. Claims-only sanity check ───────────────────────────────────────────────

def check_2_claims_only_sanity(results_dir: Path, split: str,
                                data_path: Path) -> None:
    print("\n" + "=" * 68)
    print("CHECK 2: Claims-only sanity check")
    print("=" * 68)

    # Load the dataset to inspect pairs
    recs = load_jsonl(data_path)
    eval_recs = iter_split(recs, split)
    pairs = defaultdict(list)
    for r in eval_recs:
        pairs[r.pair_id].append(r)

    # Verify claim lists are identical within pairs
    mismatch_count = 0
    for pid, members in pairs.items():
        if len(members) != 2:
            continue
        f_rec = next((r for r in members if r.condition == "faithful"), None)
        o_rec = next((r for r in members if r.condition != "faithful"), None)
        if not f_rec or not o_rec:
            continue
        f_claims = [(c.claim_id, c.text, c.is_safety_relevant) for c in f_rec.claims]
        o_claims = [(c.claim_id, c.text, c.is_safety_relevant) for c in o_rec.claims]
        if f_claims != o_claims:
            mismatch_count += 1
            print(f"  MISMATCH in pair {pid}: claim lists differ")

    if mismatch_count == 0:
        print(f"  OK: claim lists are identical for all {len(pairs)} pairs.")
        print("  The only difference between faithful and omission members")
        print("  is the policy output. Claims-only is not leaking labels.")
    else:
        print(f"  WARNING: {mismatch_count} pairs have mismatched claim lists.")

    # Show one example: what does a claims_only prompt look like for an omission?
    print("\n  SAMPLE: claims_only input for one omission case")
    print("  (confirm the safety claim IS in the list but NOT in the output)")
    sample = next((r for r in eval_recs
                   if r.condition == "omission"), None)
    if sample:
        target = next(c for c in sample.claims
                      if c.claim_id == sample.target_claim_id)
        print(f"\n  pair_id: {sample.pair_id}")
        print(f"  target claim (safety): {target.text}")
        print(f"\n  claim list shown to monitor:")
        for c in sample.claims:
            tag = "[SAFETY-RELEVANT]" if c.is_safety_relevant else "[routine]"
            print(f"    - {c.text} {tag}")
        print(f"\n  policy output (target is absent):")
        print(f"    {sample.policy_output}")
        in_output = target.summary_text.lower()[:30] in sample.policy_output.lower()
        print(f"\n  target summary text in output: {in_output}  "
              f"(should be False for omission)")

    # Load GPT-4o-mini structured claims_only results and show a few raw responses
    sc_results = load_condition(results_dir, split,
                                "structured", "claims_only", "gpt_4o_mini")
    if sc_results:
        omission_cases = [r for r in sc_results
                          if r["true_label"] == 1 and r["pred_label"] == 1][:2]
        print(f"\n  GPT-4o-mini structured claims_only — "
              f"sample raw responses on omission cases:")
        for r in omission_cases:
            print(f"\n  id: {r['id']}")
            raw = r.get("raw_response", "")
            # show first 400 chars of reasoning
            print(f"  raw (first 400 chars): {raw[:400]}")

        # Check: is the model doing genuine reasoning or just matching keywords?
        # Inspect a faithful case where it correctly said FAITHFUL
        faithful_cases = [r for r in sc_results
                          if r["true_label"] == 0 and r["pred_label"] == 0][:1]
        for r in faithful_cases:
            print(f"\n  id: {r['id']} (faithful, correctly predicted)")
            raw = r.get("raw_response", "")
            print(f"  raw (first 400 chars): {raw[:400]}")

        print(f"\n  INTERPRETATION:")
        print(f"  If the reasoning traces show the model checking each claim")
        print(f"  and noting ABSENT for the safety one, the finding is:")
        print(f"  'structured decomposition + claim list enables near-perfect")
        print(f"   detection — extraction is the bottleneck, not the format'.")
        print(f"  If it just reads the output and guesses, it is a different claim.")


# ── 3. Power analysis for test split ──────────────────────────────────────────

def check_3_power_analysis(results_dir: Path, dev_split_size: int = 48) -> None:
    print("\n" + "=" * 68)
    print("CHECK 3: Power analysis for test split")
    print("=" * 68)

    # Effect sizes observed on dev (GPT-4o-mini, the key comparisons)
    # structured_full vs holistic_full: 0.958 - 0.833 = 0.125
    # structured_output_only vs holistic_output_only: 0.813 - 0.563 = 0.250
    effect_sizes = {
        "structured_full vs holistic_full (0.125)":     0.125,
        "structured_output_only vs holistic_output_only (0.250)": 0.250,
        "holistic_full vs holistic_output_only (0.271)": 0.271,
    }

    print(f"\n  Simulation: paired bootstrap power at various n")
    print(f"  (1000 simulated experiments per cell, alpha=0.05)")
    print(f"  Effect sizes observed on dev (GPT-4o-mini):\n")
    print(f"  {'effect':50s} {'n=24':>6s} {'n=48':>6s} {'n=96':>6s} {'n=192':>6s}")
    print(f"  {'-'*50} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    rng = np.random.RandomState(42)
    for label, effect in effect_sizes.items():
        row = []
        for n in [24, 48, 96, 192]:
            # simulate n paired examples where A = B + effect on average,
            # with realistic variance (std ~ 0.45 for balanced binary outcomes)
            n_sig = 0
            for _ in range(1000):
                # true labels: balanced
                y_true = np.array([0]*( n//2) + [1]*(n//2))
                # A predictions: better monitor, BA ~ 0.833 + effect
                # B predictions: weaker monitor, BA ~ 0.833
                # simulate by flipping labels with error rates
                err_b = (1 - 0.833) / 2   # ~8.4% error per class
                err_a = max(0, err_b - effect / 2)
                y_b = y_true.copy()
                y_a = y_true.copy()
                for i in range(n):
                    if rng.random() < err_b:
                        y_b[i] = 1 - y_b[i]
                    if rng.random() < err_a:
                        y_a[i] = 1 - y_a[i]
                ra = [{"id": str(i), "true_label": int(y_true[i]),
                       "pred_label": int(y_a[i])} for i in range(n)]
                rb = [{"id": str(i), "true_label": int(y_true[i]),
                       "pred_label": int(y_b[i])} for i in range(n)]
                comp = paired_bootstrap_comparison(ra, rb, seed=int(rng.randint(1000)),
                                                   n_boot=500)
                if comp.get("ci_lo", 0) > 0:
                    n_sig += 1
            row.append(f"{n_sig/10:5.1f}%")
        print(f"  {label:50s} {'  '.join(row)}")

    print(f"\n  Your test split has n=48 (24 pairs × 2 conditions).")
    print(f"  For the 0.125 effect (structured vs holistic full), power at n=48")
    print(f"  is the value above — check whether it exceeds 80%.")
    print(f"  For the 0.250 effect (output_only gap), power should be higher.")
    print(f"  If power < 80% at n=48: generate additional held-out test pairs")
    print(f"  (never touched by prompt design) before the final run.")
    print(f"  Recommended minimum: n=96 if the 0.125 effect is underpowered.")


# ── 4. Headline stability note ────────────────────────────────────────────────

def check_4_stability_note() -> None:
    print("\n" + "=" * 68)
    print("CHECK 4: Stability check instructions")
    print("=" * 68)
    print("""
  The 1.000 BA for GPT-4o-mini structured claims_only needs one re-run
  with a different seed and different few-shot examples before trusting it.

  Run this:
    python eval/run.py --split dev --model gpt-4o-mini --shots 3 \\
      --styles structured --levels claims_only --seed 99

  What to check:
    - Does BA stay >= 0.95? (ceiling effect is real, small variance expected)
    - Does kappa stay >= 0.90?
    - If it drops below 0.90, the 1.000 was a lucky few-shot draw.
      Report the mean of the two runs, not the max.

  Also re-run holistic claims_only (0.979) with seed 99:
    python eval/run.py --split dev --model gpt-4o-mini --shots 3 \\
      --styles holistic --levels claims_only --seed 99

  These two re-runs cost < $0.02 total.
""")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--split",   default="dev")
    ap.add_argument("--data",    default="data/dataset.jsonl")
    ap.add_argument("--checks",  nargs="+", default=["1","2","3","4"],
                    choices=["1","2","3","4"],
                    help="Which checks to run (default: all)")
    args = ap.parse_args()

    results_dir = ROOT / args.results
    data_path   = ROOT / args.data
    conds = load_all_conditions(results_dir, args.split)

    if "1" in args.checks:
        check_1_bootstrap_gaps(conds)
    if "2" in args.checks:
        check_2_claims_only_sanity(results_dir, args.split, data_path)
    if "3" in args.checks:
        check_3_power_analysis(results_dir, dev_split_size=48)
    if "4" in args.checks:
        check_4_stability_note()

    print("\n" + "=" * 68)
    print("NEXT ACTIONS")
    print("=" * 68)
    print("""
  After reviewing the output above:

  A. If power at n=48 test is < 80% for the 0.125 effect:
       python data/generate.py --n 200 --out data --seed 7
     This regenerates with more test pairs. Verify splits.json shows
     test n >= 96. Then re-run prompt design iteration on dev only.

  B. Stability re-runs (seed 99):
       python eval/run.py --split dev --model gpt-4o-mini --shots 3 \\
         --styles structured holistic --levels claims_only --seed 99

  C. GPT-4o on test (deferred until final run):
       python eval/run.py --split test --model gpt-4o --shots 3 \\
         --i_have_read_prereg

  D. Final test run (once, all models):
       python eval/run.py --split test \\
         --model llama3:latest gpt-4o-mini gpt-4o --shots 3 \\
         --i_have_read_prereg
""")


if __name__ == "__main__":
    main()