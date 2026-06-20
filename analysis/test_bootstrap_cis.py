"""
analysis/test_bootstrap_cis.py — paired bootstrap CIs on all pre-registered
test comparisons. Run from repo root:

  python analysis/test_bootstrap_cis.py
"""

from __future__ import annotations
import json, sys, numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def load(split, style, level, model_tag):
    for f in sorted(RESULTS.glob(f"{split}_{style}_{level}_{model_tag}*.json")):
        d = json.loads(f.read_text())
        if "results" in d:
            return [r for r in d["results"] if r.get("pred_label") is not None]
    return None


def paired_bootstrap(ra, rb, n_boot=5000, seed=42):
    np.random.seed(seed)
    ids_a = {r["id"]: r for r in ra}
    ids_b = {r["id"]: r for r in rb}
    shared = sorted(set(ids_a) & set(ids_b))
    if len(shared) < 5:
        return None
    yt = np.array([ids_a[i]["true_label"] for i in shared])
    ya = np.array([ids_a[i]["pred_label"] for i in shared])
    yb = np.array([ids_b[i]["pred_label"] for i in shared])
    obs_a = balanced_accuracy_score(yt, ya)
    obs_b = balanced_accuracy_score(yt, yb)
    obs_d = obs_a - obs_b
    diffs = []
    n = len(shared)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        yt_ = yt[idx]
        if len(np.unique(yt_)) < 2:
            continue
        diffs.append(balanced_accuracy_score(yt_, ya[idx]) -
                     balanced_accuracy_score(yt_, yb[idx]))
    diffs = np.array(diffs)
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    p  = float(np.mean(diffs <= 0))
    if lo > 0:
        verdict = "SIGNIFICANT (CI excludes zero, A > B)"
    elif hi < 0:
        verdict = "SIGNIFICANT (CI excludes zero, B > A)"
    else:
        verdict = "null (CI includes zero)"
    return {
        "n": len(shared), "A": round(obs_a, 4), "B": round(obs_b, 4),
        "diff": round(obs_d, 4), "lo": round(lo, 4), "hi": round(hi, 4),
        "p": round(p, 4), "verdict": verdict,
    }


def pr(label, r):
    if r is None:
        print(f"  {label:55s}  MISSING DATA")
        return
    sig = "*" if "SIGNIFICANT" in r["verdict"] else " "
    print(f"  {label:55s}  "
          f"A={r['A']:.3f} B={r['B']:.3f} diff={r['diff']:+.3f} "
          f"[{r['lo']:+.3f},{r['hi']:+.3f}] p={r['p']:.3f} {sig} {r['verdict']}")


def main():
    split = "test"
    models = [
        ("llama3:latest",  "llama3_latest"),
        ("gpt-4o-mini",    "gpt_4o_mini"),
        ("gpt-4o",         "gpt_4o"),
    ]

    print("=" * 100)
    print("PAIRED BOOTSTRAP CIs — TEST SPLIT (n=120, 5000 resamples)")
    print("=" * 100)

    print("\n--- H1: structured vs holistic at full source ---")
    for model, tag in models:
        s = load(split, "structured", "full", tag)
        h = load(split, "holistic",   "full", tag)
        r = paired_bootstrap(s, h) if s and h else None
        pr(f"{model}: structured_full vs holistic_full", r)

    print("\n--- C2: structured vs holistic at output_only ---")
    for model, tag in models:
        s = load(split, "structured", "output_only", tag)
        h = load(split, "holistic",   "output_only", tag)
        r = paired_bootstrap(s, h) if s and h else None
        pr(f"{model}: structured_output_only vs holistic_output_only", r)

    print("\n--- C3: holistic degradation full → output_only ---")
    for model, tag in models:
        f = load(split, "holistic", "full",       tag)
        o = load(split, "holistic", "output_only", tag)
        r = paired_bootstrap(f, o) if f and o else None
        pr(f"{model}: holistic full vs output_only", r)

    print("\n--- C3b: structured degradation full → output_only ---")
    for model, tag in models:
        f = load(split, "structured", "full",       tag)
        o = load(split, "structured", "output_only", tag)
        r = paired_bootstrap(f, o) if f and o else None
        pr(f"{model}: structured full vs output_only", r)

    print("\n--- C4: claims_only_untagged vs full ---")
    for model, tag in models:
        for style in ["holistic", "structured"]:
            c = load(split, style, "claims_only_untagged", tag)
            f = load(split, style, "full", tag)
            r = paired_bootstrap(c, f) if c and f else None
            pr(f"{model} {style}: claims_only_untagged vs full", r)

    print("\n--- GPT-4o reversal check: holistic vs structured at output_only ---")
    s = load(split, "structured", "output_only", "gpt_4o")
    h = load(split, "holistic",   "output_only", "gpt_4o")
    r = paired_bootstrap(h, s) if s and h else None
    pr("gpt-4o: holistic_output_only vs structured_output_only", r)
    if r:
        print(f"\n  GPT-4o reversal is {'a TIE (not significant)' if 'null' in r['verdict'] else 'SIGNIFICANT — report direction'}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()