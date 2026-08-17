import json, glob

# holistic output-only result files (the below-chance inversion cells)
files = glob.glob("results/*holistic_output_only*.json")
for f in files:
    print("\n" + "#"*78)
    print("#", f)
    print("#"*78)
    data = json.load(open(f))
    rows = data if isinstance(data, list) else data.get("predictions", data.get("results", []))
    shown = 0
    for r in rows:
        if not isinstance(r, dict): 
            continue
        true = r.get("true_label"); pred = r.get("pred_label")
        raw = r.get("raw_response","")
        # show cases where the model got it wrong, to inspect the inversion
        if true is not None and pred is not None and true != pred:
            print(f"\n[true={true} pred={pred}] {raw[:400]}")
            shown += 1
        if shown >= 8:
            break
