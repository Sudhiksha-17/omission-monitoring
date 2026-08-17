import sys
from pathlib import Path
sys.path.insert(0, str(Path("data")))
from schema import load_jsonl
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

recs = list(load_jsonl("data/dataset.jsonl"))
train = [r for r in recs if r.split == "train"]
test  = [r for r in recs if r.split == "test"]

def probe(get_text, label):
    Xtr = [get_text(r) for r in train]; ytr = np.array([r.label for r in train])
    Xte = [get_text(r) for r in test];  yte = np.array([r.label for r in test])
    vec = CountVectorizer(ngram_range=(1,2), min_df=2, lowercase=True)
    Xtr_v = vec.fit_transform(Xtr); Xte_v = vec.transform(Xte)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    clf.fit(Xtr_v, ytr)
    ba = balanced_accuracy_score(yte, clf.predict(Xte_v))
    print("  %-32s TEST bag-of-words BA = %.4f  (vocab %d)" % (label, ba, len(vec.vocabulary_)))

print("="*64)
print("WHERE DOES THE LEAK LIVE? bag-of-words BA per text field")
print("="*64)
probe(lambda r: r.policy_output, "policy_output (monitor input)")
probe(lambda r: r.source_text,   "source_text")
probe(lambda r: " ".join(c.text for c in r.claims), "claims (concatenated)")
print()
print("  0.50 = no leak (chance). 1.00 = perfectly separable.")
print()

print("="*64)
print("MATCHED PAIRS (same pair_id, both classes)")
print("="*64)
from collections import defaultdict
by_pair = defaultdict(dict)
for r in test:
    by_pair[r.pair_id][r.label] = r
shown = 0
for pid, d in by_pair.items():
    if 0 in d and 1 in d:
        om = d[1]; fa = d[0]
        print("\n--- pair_id %s ---" % pid)
        print("[OMISSION label 1]:\n  " + om.policy_output)
        print("[FAITHFUL label 0]:\n  " + fa.policy_output)
        shown += 1
    if shown >= 3:
        break
