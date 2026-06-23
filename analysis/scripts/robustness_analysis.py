"""
analysis/robustness_analysis.py — analyze robustness_dev.jsonl.

For each model × style × variant, computes BA, sensitivity, specificity.
Then checks whether the structured-beats-holistic-on-specificity finding
survives prompt paraphrase (v0, v1, v2).

Run from repo root:
  python analysis/robustness_analysis.py
"""

from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

ROOT   = Path(__file__).parent.parent
JSONL  = Path(__file__).parent / "robustness_dev.jsonl"


def load() -> dict:
    """Load results grouped by (model, style, variant)."""
    groups: dict[tuple, list] = defaultdict(list)
    with JSONL.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("pred_label") is not None and not r.get("api_error", False):
                key = (r["model"], r["style"], r["variant"])
                groups[key].append(r)
    return groups


def metrics(records: list) -> dict:
    yt = np.array([r["true_label"] for r in records])
    yp = np.array([r["pred_label"] for r in records])
    ba   = balanced_accuracy_score(yt, yp)
    sens = float((yp[yt == 1] == 1).mean()) if (yt == 1).sum() > 0 else float("nan")
    spec = float((yp[yt == 0] == 0).mean()) if (yt == 0).sum() > 0 else float("nan")
    kappa = cohen_kappa_score(yt, yp)
    return {"n": len(records), "ba": round(ba, 3), "sens": round(sens, 3),
            "spec": round(spec, 3), "kappa": round(kappa, 3)}


def main():
    groups = load()
    if not groups:
        print(f"No data found in {JSONL}")
        return

    models   = sorted({k[0] for k in groups})
    styles   = ["holistic", "structured"]
    variants = ["v0", "v1", "v2"]

    print("=" * 80)
    print("ROBUSTNESS CHECK — dev split, full source, 3 prompt variants")
    print("Claim to test: structured significantly higher specificity than holistic")
    print("across all variants (v0=original, v1=paraphrase-1, v2=paraphrase-2)")
    print("=" * 80)

    print(f"\n  {'model':25s} {'style':12s} {'variant':8s} {'BA':>6s} {'sens':>6s} {'spec':>6s} {'kappa':>7s} {'n':>5s}")
    print(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*5}")

    # track specificity per model×variant for the robustness verdict
    spec_by_model_variant: dict[tuple, dict] = defaultdict(dict)

    for model in models:
        for variant in variants:
            for style in styles:
                key   = (model, style, variant)
                recs  = groups.get(key, [])
                if not recs:
                    print(f"  {model:25s} {style:12s} {variant:8s}  MISSING")
                    continue
                m = metrics(recs)
                print(f"  {model:25s} {style:12s} {variant:8s} "
                      f"{m['ba']:6.3f} {m['sens']:6.3f} {m['spec']:6.3f} "
                      f"{m['kappa']:7.3f} {m['n']:5d}")
                spec_by_model_variant[(model, variant)][style] = m["spec"]
        print()

    # Robustness verdict
    print("=" * 80)
    print("SPECIFICITY ADVANTAGE (structured - holistic) per model × variant")
    print("Claim holds if structured spec > holistic spec across all variants")
    print("=" * 80)
    print(f"\n  {'model':25s} {'variant':8s} {'S_spec':>7s} {'H_spec':>7s} {'diff':>7s} {'holds':>6s}")
    print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

    all_hold = True
    for model in models:
        for variant in variants:
            sv = spec_by_model_variant.get((model, variant), {})
            s_spec = sv.get("structured", float("nan"))
            h_spec = sv.get("holistic",   float("nan"))
            if s_spec != s_spec or h_spec != h_spec:  # nan check
                print(f"  {model:25s} {variant:8s}  MISSING")
                continue
            diff  = s_spec - h_spec
            holds = diff > 0
            if not holds:
                all_hold = False
            mark = "YES" if holds else "NO *"
            print(f"  {model:25s} {variant:8s} {s_spec:7.3f} {h_spec:7.3f} "
                  f"{diff:+7.3f} {mark:>6s}")
        print()

    print("=" * 80)
    if all_hold:
        print("VERDICT: ROBUST — structured specificity > holistic specificity")
        print("across all models and all prompt variants.")
        print("The specificity advantage is a property of the decomposition paradigm,")
        print("not an artifact of the specific prompt wording.")
    else:
        print("VERDICT: NOT FULLY ROBUST — at least one variant shows holistic >= structured.")
        print("The specificity advantage is partially prompt-dependent.")
        print("Report which variants hold and which do not.")
    print("=" * 80)

    # Also report BA robustness
    print("\nBA ADVANTAGE (structured - holistic) per model × variant")
    print(f"\n  {'model':25s} {'variant':8s} {'S_BA':>6s} {'H_BA':>6s} {'diff':>7s}")
    print(f"  {'-'*25} {'-'*8} {'-'*6} {'-'*6} {'-'*7}")

    ba_by_mv: dict[tuple, dict] = defaultdict(dict)
    for model in models:
        for variant in variants:
            for style in styles:
                key  = (model, style, variant)
                recs = groups.get(key, [])
                if recs:
                    ba_by_mv[(model, variant)][style] = metrics(recs)["ba"]
    for model in models:
        for variant in variants:
            bv = ba_by_mv.get((model, variant), {})
            s_ba = bv.get("structured", float("nan"))
            h_ba = bv.get("holistic",   float("nan"))
            if s_ba != s_ba or h_ba != h_ba:
                continue
            print(f"  {model:25s} {variant:8s} {s_ba:6.3f} {h_ba:6.3f} {s_ba-h_ba:+7.3f}")
        print()


if __name__ == "__main__":
    main()