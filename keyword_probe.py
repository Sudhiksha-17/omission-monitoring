import sys
from pathlib import Path
sys.path.insert(0, str(Path("data")))
from schema import load_jsonl

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

recs = list(load_jsonl("data/dataset.jsonl"))
train = [r for r in recs if r.split == "train"]
dev   = [r for r in recs if r.split == "dev"]
test  = [r for r in recs if r.split == "test"]

def texts_labels(rs):
    return [r.policy_output for r in rs], np.array([r.label for r in rs])

Xtr_t, ytr = texts_labels(train)
Xdv_t, ydv = texts_labels(dev)
Xte_t, yte = texts_labels(test)

# bag-of-words on the POLICY OUTPUT only (same input the monitor sees at output_only).
# unigrams + bigrams, drop very rare terms. This is the lexical-shortcut probe:
# can safety vocabulary alone separate the classes?
vec = CountVectorizer(ngram_range=(1,2), min_df=2, lowercase=True)
Xtr = vec.fit_transform(Xtr_t)
Xdv = vec.transform(Xdv_t)
Xte = vec.transform(Xte_t)

best_c, best_dv = 1.0, -1.0
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    clf = LogisticRegression(C=C, max_iter=2000, random_state=42)
    clf.fit(Xtr, ytr)
    ba = balanced_accuracy_score(ydv, clf.predict(Xdv))
    if ba > best_dv:
        best_dv, best_c = ba, C

clf = LogisticRegression(C=best_c, max_iter=2000, random_state=42)
clf.fit(Xtr, ytr)
yp = clf.predict(Xte)
ba_test = balanced_accuracy_score(yte, yp)
kap = cohen_kappa_score(yte, yp)
maj = int(np.bincount(ytr).argmax())
ba_maj = balanced_accuracy_score(yte, np.full_like(yte, maj))

print("="*60)
print("KEYWORD / BAG-OF-WORDS PROBE (policy output only)")
print("="*60)
print(f"  vocab size:            {len(vec.vocabulary_)}")
print(f"  best C (dev):          {best_c}")
print(f"  dev balanced accuracy: {best_dv:.4f}")
print(f"  TEST balanced accuracy:{ba_test:.4f}")
print(f"  test kappa:            {kap:.4f}")
print(f"  majority baseline BA:  {ba_maj:.4f}")
print()
print(f"  (compare: your length+sentiment probe was 0.558 dev)")
print()

# show the most class-discriminative tokens, so you can see WHAT it keys on
import numpy as np
coefs = clf.coef_[0]
names = np.array(vec.get_feature_names_out())
order = np.argsort(coefs)
print("  Top tokens -> OMISSION (label 1):")
for i in order[-15:][::-1]:
    print(f"     {coefs[i]:+.2f}  {names[i]}")
print("  Top tokens -> FAITHFUL (label 0):")
for i in order[:15]:
    print(f"     {coefs[i]:+.2f}  {names[i]}")
