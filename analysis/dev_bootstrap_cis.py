"""
analysis/dev_bootstrap_cis.py — paired bootstrap CIs on all key contrasts
from the dev split. Run before the frozen test.

Primary pre-registered comparison: H1 — structured vs holistic BA at full source.
All other comparisons are exploratory.

Run from repo root:
  python analysis/dev_bootstrap_cis.py
"""

from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from metrics import paired_bootstrap_comparison, dissociation_test

RESULTS = ROOT / "results"


def load(split, style, level, model_tag):
    # Try proper named files first
    for f in sorted(RESULTS.glob(f"{split}_{style}_{level}_{model_tag}*.json"),
                    reverse=True):
        d = json.loads(f.read_text())
        if "results" in d:
            return [r for r in d["results"] if r.get("pred_label") is not None]
    # Try legacy gpt_oss files
    for f in sorted(RESULTS.glob(f"{split}_{style}_{level}_*{model_tag}*.json"),
                    reverse=True):
        d = json.loads(f.read_text())
        if "results" in d:
            return [r for r in d["results"] if r.get("pred_label") is not None]
    # Read from any file with matching metadata
    for f in sorted(RESULTS.glob("*.json"), reverse=True):
        try:
            d = json.loads(f.read_text())
            if "results" not in d:
                continue
            recs = d["results"]
            if not recs:
                continue
            r0 = recs[0]
            if (r0.get("split") == split and
                    r0.get("style") == style and
                    r0.get("source_level") == level and
                    model_tag.replace("_", "/") in r0.get("model", "") or
                    model_tag in r0.get("model", "").replace("/", "_").replace("-", "_")):
                return [r for r in recs if r.get("pred_label") is not None]
        except Exception:
            continue
    return None


def pr(label, r):
    if r is None or "error" in r:
        print(f"  {label}: MISSING")
        return
    sig = "*" if "CI excludes zero" in r.get("verdict", "") else " "
    print(f"  {label}")
    print(f"    {r['metric']:12s} A={r['structured']:.3f} B={r['holistic']:.3f} "
          f"diff={r['observed_diff']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
          f"p={r['p_value']:.3f} {sig} {r['verdict']}")


def main():
    split = "dev"
    models = [
        ("gpt-4o-mini",         "gpt_4o_mini"),
        ("openai/gpt-oss-20b",  "openai_gpt_oss_20b"),
        ("openai/gpt-oss-120b", "openai_gpt_oss_120b"),
    ]

    print("=" * 80)
    print("DEV PAIRED BOOTSTRAP CIs (5000 resamples)")
    print("PRIMARY: H1 = structured vs holistic BA at full source")
    print("All other comparisons are EXPLORATORY")
    print("=" * 80)

    for model_display, model_tag in models:
        s_full   = load(split, "structured", "full",                model_tag)
        h_full   = load(split, "holistic",   "full",                model_tag)
        s_claims = load(split, "structured", "claims_only_untagged", model_tag)
        h_claims = load(split, "holistic",   "claims_only_untagged", model_tag)
        s_out    = load(split, "structured", "output_only",          model_tag)
        h_out    = load(split, "holistic",   "output_only",          model_tag)

        print(f"\n{'='*80}")
        print(f"  MODEL: {model_display}")
        print(f"{'='*80}")

        # H1 — primary pre-registered
        if s_full and h_full:
            print(f"\n  [PRIMARY] H1: structured vs holistic at full source")
            for metric in ("ba", "sensitivity", "specificity"):
                r = paired_bootstrap_comparison(s_full, h_full, metric=metric,
                                                n_boot=5000, seed=42, silent=True)
                r["structured"] = r.get("structured", r.get("A", 0))
                r["holistic"]   = r.get("holistic",   r.get("B", 0))
                pr(f"  H1 {metric}", r)

        # Dissociation at each level
        for level, sr, hr in [
            ("full",                s_full,   h_full),
            ("claims_only_untagged", s_claims, h_claims),
            ("output_only",          s_out,    h_out),
        ]:
            if sr and hr:
                print(f"\n  [EXPLORATORY] Dissociation test at {level}")
                dt = dissociation_test(sr, hr, level=level, n_boot=5000, seed=42)

        # Source-access degradation
        if s_full and s_out:
            print(f"\n  [EXPLORATORY] Structured: full vs output_only degradation")
            r = paired_bootstrap_comparison(s_full, s_out, metric="ba",
                                            n_boot=5000, seed=42, silent=True)
            diff   = r.get("observed_diff", 0)
            lo, hi = r.get("ci_lo", 0), r.get("ci_hi", 0)
            sig    = "*" if lo > 0 else " "
            print(f"    structured full vs output_only: "
                  f"full={r.get('structured',0):.3f} out={r.get('holistic',0):.3f} "
                  f"diff={diff:+.3f} [{lo:+.3f},{hi:+.3f}] {sig} "
                  f"{'SIGNIFICANT degradation' if lo > 0 else 'null'}")

        if h_full and h_out:
            print(f"\n  [EXPLORATORY] Holistic: full vs output_only degradation")
            r = paired_bootstrap_comparison(h_full, h_out, metric="ba",
                                            n_boot=5000, seed=42, silent=True)
            diff   = r.get("observed_diff", 0)
            lo, hi = r.get("ci_lo", 0), r.get("ci_hi", 0)
            sig    = "*" if lo > 0 else " "
            print(f"    holistic full vs output_only: "
                  f"full={r.get('structured',0):.3f} out={r.get('holistic',0):.3f} "
                  f"diff={diff:+.3f} [{lo:+.3f},{hi:+.3f}] {sig} "
                  f"{'SIGNIFICANT degradation' if lo > 0 else 'null'}")

    print("\n" + "=" * 80)
    print("DONE. Lock these CIs before running the frozen test.")
    print("=" * 80)


if __name__ == "__main__":
    main()