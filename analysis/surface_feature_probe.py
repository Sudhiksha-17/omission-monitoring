"""
analysis/surface_feature_probe.py — surface feature check on claims.

Tests whether safety-relevant claims in the synthetic dataset are
separable from routine claims on cheap surface features alone:
character length, word count, and a simple sentiment/valence proxy
(count of negative-affect words).

If a logistic regression on these features achieves high accuracy,
then the monitor may be detecting "a vivid/negative claim is missing"
rather than "a safety-relevant claim is missing." This is the
validity threat named by the hostile reviewer.

Run from repo root:
  python analysis/surface_feature_probe.py

No API calls. Runs in seconds.
"""

from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "data"))
from schema import load_jsonl


# Negative-affect / hazard word list — deliberately coarse,
# the kind a trivial classifier might exploit.
NEGATIVE_WORDS = {
    "fail", "failure", "error", "risk", "hazard", "danger", "unsafe",
    "harm", "injur", "damage", "breach", "vulnerab", "incident",
    "adverse", "critical", "severe", "serious", "fatal", "toxic",
    "leak", "crack", "break", "fault", "defect", "flaw", "bug",
    "attack", "exploit", "malicious", "corrupt", "loss", "lost",
    "overheat", "shock", "fire", "burn", "collaps", "crash", "halt",
    "wrong", "incorrect", "missing", "absent", "omit", "suppress",
    "conceal", "hide", "withhold", "deny", "false", "fraudul",
    "withdraw", "recall", "alert", "warn", "caution", "prohibit",
}


def extract_claim_features(text: str) -> list[float]:
    words = text.lower().split()
    char_len    = len(text)
    word_count  = len(words)
    neg_count   = sum(1 for w in words
                      if any(w.startswith(stem) for stem in NEGATIVE_WORDS))
    neg_density = neg_count / max(word_count, 1)
    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
    # punctuation density (exclamation, colons often signal warnings)
    punct_count = sum(1 for c in text if c in "!:;")
    return [char_len, word_count, neg_count, neg_density,
            avg_word_len, punct_count]


def main():
    recs = load_jsonl(ROOT / "data" / "dataset.jsonl")

    # Collect all claims across all records (deduplicate by text)
    seen = set()
    claims_data = []
    for rec in recs:
        for c in rec.claims:
            if c.text in seen:
                continue
            seen.add(c.text)
            feats = extract_claim_features(c.text)
            claims_data.append({
                "text": c.text,
                "is_safety": int(c.is_safety_relevant),
                "features": feats,
            })

    X = np.array([d["features"] for d in claims_data])
    y = np.array([d["is_safety"] for d in claims_data])

    print(f"Total unique claims: {len(claims_data)}")
    print(f"Safety-relevant: {y.sum()}  Routine: {(1-y).sum()}")
    print(f"Balance: {y.mean():.3f} positive rate")

    # Feature means by class
    feat_names = ["char_len", "word_count", "neg_count",
                  "neg_density", "avg_word_len", "punct_count"]
    print("\nFeature means by class:")
    print(f"  {'feature':15s} {'routine':>10s} {'safety':>10s} {'ratio':>8s}")
    for i, name in enumerate(feat_names):
        routine_mean = X[y==0, i].mean()
        safety_mean  = X[y==1, i].mean()
        ratio = safety_mean / max(routine_mean, 1e-9)
        print(f"  {name:15s} {routine_mean:10.3f} {safety_mean:10.3f} {ratio:8.2f}x")

    # Logistic regression with 5-fold CV
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    cv_bas = cross_val_score(clf, X, y,
                             cv=5, scoring="balanced_accuracy")
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_ba = balanced_accuracy_score(y, y_pred)

    print(f"\nLogistic regression on surface features (5-fold CV):")
    print(f"  CV balanced accuracy: {cv_bas.mean():.3f} ± {cv_bas.std():.3f}")
    print(f"  Train balanced accuracy: {train_ba:.3f}")
    print()
    print(classification_report(y, y_pred,
                                 target_names=["routine", "safety"],
                                 zero_division=0))

    # Verdict
    cv_ba = cv_bas.mean()
    print("=" * 60)
    if cv_ba < 0.65:
        print(f"OK: Surface features do not reliably separate safety from")
        print(f"routine claims (CV BA={cv_ba:.3f} < 0.65).")
        print(f"The monitor is unlikely to be detecting surface differences")
        print(f"rather than genuine safety relevance.")
        print(f"Disclose this check and report the CV BA in your methods.")
    elif cv_ba < 0.80:
        print(f"MODERATE CONCERN: Surface features achieve CV BA={cv_ba:.3f}.")
        print(f"Safety claims are somewhat separable on surface features.")
        print(f"Disclose this and note it as a validity limitation.")
        print(f"Consider counterbalancing with routine-claim omissions.")
    else:
        print(f"HIGH CONCERN: Surface features achieve CV BA={cv_ba:.3f}.")
        print(f"Safety claims are strongly separable from routine claims")
        print(f"on cheap features. The monitor may be detecting surface")
        print(f"differences, not safety relevance. This needs to be addressed")
        print(f"before publication.")
    print("=" * 60)

    # Most discriminative examples (top safety vs routine by neg_density)
    print("\nTop 3 most negative-dense safety claims:")
    safety_claims = [d for d in claims_data if d["is_safety"]]
    safety_claims.sort(key=lambda d: d["features"][3], reverse=True)
    for d in safety_claims[:3]:
        print(f"  neg_density={d['features'][3]:.3f}: {d['text'][:90]}")

    print("\nTop 3 most negative-dense ROUTINE claims (should be low):")
    routine_claims = [d for d in claims_data if not d["is_safety"]]
    routine_claims.sort(key=lambda d: d["features"][3], reverse=True)
    for d in routine_claims[:3]:
        print(f"  neg_density={d['features'][3]:.3f}: {d['text'][:90]}")


if __name__ == "__main__":
    main()