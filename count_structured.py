import json, glob, os

# structured output-only, all models, dev and test
files = sorted(glob.glob("results/*structured_output_only*.json"))
for f in files:
    d = json.load(open(f))
    rows = d if isinstance(d, list) else d.get("predictions", d.get("results", []))
    pairs = [(r.get("true_label"), r.get("pred_label")) for r in rows
             if isinstance(r, dict) and r.get("pred_label") is not None]
    n = len(pairs)
    if n == 0:
        print(f"{os.path.basename(f)}: no parseable predictions"); continue
    preds = [p for _, p in pairs]; trues = [t for t, _ in pairs]
    om = sum(1 for p in preds if p == 1); fa = sum(1 for p in preds if p == 0)
    t1 = [i for i in range(n) if trues[i] == 1]; t0 = [i for i in range(n) if trues[i] == 0]
    sens = sum(1 for i in t1 if preds[i] == 1)/len(t1) if t1 else float("nan")
    spec = sum(1 for i in t0 if preds[i] == 0)/len(t0) if t0 else float("nan")
    print(f"{os.path.basename(f)}")
    print(f"   n={n}  pred OMISSION={om} ({om/n:.0%})  pred FAITHFUL={fa} ({fa/n:.0%})")
    print(f"   sensitivity={sens:.2f}  specificity={spec:.2f}")
    print()
