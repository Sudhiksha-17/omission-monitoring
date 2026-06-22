"""
analysis/calibration.py — Phase 2 analysis script.

Produces two outputs the writeup needs:

Table 1: Calibration decomposition (H1 story)
  For each (model, style, level): sensitivity (omission recall),
  specificity (faithful recall), BA, kappa, and FPR.
  The key claim to verify: holistic_full has high sensitivity but low
  specificity; structured_full restores specificity without sacrificing
  sensitivity. If this survives at n>=200 with tight CIs, it is the
  clean mechanism claim.

Table 2: Source-access effect (novel condition)
  For holistic and structured separately: BA and specificity as a function
  of source level (full → claims_only → output_only).
  The key question: does degrading source access primarily destroy
  specificity (monitor loses reference, over-flags) or sensitivity
  (monitor can no longer detect absence), and is the degradation larger
  than the sabotage-monitoring less-is-more literature would predict?

Run:
  python analysis/calibration.py --results results/ --split dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from metrics import compute_metrics


def load_results(results_dir: Path, split: str) -> list[dict]:
    """Load all per-condition result files for a given split."""
    all_res = []
    # Catch both properly-named files and legacy files without split prefix
    candidates = sorted(set(
        list(results_dir.glob("*.json")) +
        list(results_dir.glob(f"{split}_*.json"))
    ))
    for f in candidates:
        if any(x in f.name for x in ("baseline", "paired", "H1", "dissociation")):
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if "results" not in data or "metrics" not in data:
            continue
        results_list = data.get("results", [])
        if not results_list:
            continue
        # Check this file belongs to the requested split
        if results_list[0].get("split") != split:
            continue
        cond = infer_condition(f.name, data)
        if not cond or cond.get("style") not in ("holistic", "structured"):
            continue
        data["_condition"] = cond
        all_res.append(data)
    return all_res


def calibration_row(results: list[dict], m: dict) -> dict:
    """Extract sensitivity, specificity, FPR from a compute_metrics result."""
    pc = m.get("per_class", {})
    sensitivity  = pc.get("omission",  {}).get("recall", float("nan"))
    specificity  = pc.get("faithful",  {}).get("recall", float("nan"))
    fpr = 1.0 - specificity
    return {
        "ba":          m["balanced_accuracy"],
        "kappa":       m["kappa"],
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "fpr":         round(fpr, 4),
        "n":           m["n_parseable"],
    }


def infer_condition(filename: str, data: dict) -> dict:
    """Parse model, style, level from result file — reads from JSON content first."""
    # Primary: read from the results list (every record has style, source_level, model)
    results = data.get("results", [])
    if results:
        r = results[0]
        model = r.get("model", "")
        style = r.get("style", "")
        level = r.get("source_level", "")
        if model and style and level:
            # normalize level display name
            if level == "claims_only_untagged":
                level = "claims_only*"
            return {"style": style, "level": level, "model": model}

    # Fallback: parse from filename
    # filename format: {split}_{style}_{level}_{model}_{ts}.json
    parts = filename.replace(".json", "").split("_")
    if len(parts) < 4:
        return {}
    style = parts[1] if parts[1] in ("holistic", "structured") else "unknown"
    level_raw = parts[2]
    if level_raw == "full":
        level = "full"
    elif level_raw == "claims":
        level = "claims_only*"
    elif level_raw == "output":
        level = "output_only"
    else:
        level = level_raw
    model_parts = parts[3:-1]
    model = "_".join(model_parts).replace("_latest", ":latest").replace("gpt_", "gpt-")
    return {"style": style, "level": level, "model": model}


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{'='*72}")
    print(title)
    print(f"{'='*72}")
    print(f"  {'model':20s} {'style':12s} {'level':14s} "
          f"{'BA':>6s} {'kappa':>6s} {'sens':>6s} {'spec':>6s} {'FPR':>6s} {'n':>5s}")
    print(f"  {'-'*20} {'-'*12} {'-'*14} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
    for r in rows:
        print(f"  {r['model']:20s} {r['style']:12s} {r['level']:14s} "
              f"{r['ba']:6.3f} {r['kappa']:6.3f} "
              f"{r['sensitivity']:6.3f} {r['specificity']:6.3f} "
              f"{r['fpr']:6.3f} {r['n']:5d}")


def source_access_effect(rows: list[dict]) -> None:
    """
    Print the source-access degradation table for each (model, style).
    The key comparison: full → output_only degradation in specificity vs sensitivity.
    """
    print(f"\n{'='*72}")
    print("SOURCE-ACCESS EFFECT: degradation from full to output_only")
    print(f"{'='*72}")
    print(f"  {'model':20s} {'style':12s}  "
          f"{'Δ BA':>8s} {'Δ sens':>8s} {'Δ spec':>8s}  "
          f"{'interpretation':30s}")
    print(f"  {'-'*20} {'-'*12}  {'-'*8} {'-'*8} {'-'*8}  {'-'*30}")

    by_key: dict[tuple, dict] = {}
    for r in rows:
        key = (r["model"], r["style"], r["level"])
        by_key[key] = r

    models = sorted({r["model"] for r in rows})
    styles = ["holistic", "structured"]
    for model in models:
        for style in styles:
            full  = by_key.get((model, style, "full"))
            outonly = by_key.get((model, style, "output_only"))
            if not full or not outonly:
                continue
            d_ba   = outonly["ba"]          - full["ba"]
            d_sens = outonly["sensitivity"] - full["sensitivity"]
            d_spec = outonly["specificity"] - full["specificity"]

            if abs(d_spec) > abs(d_sens) and d_spec < 0:
                interp = "specificity-dominant degradation"
            elif abs(d_sens) > abs(d_spec) and d_sens < 0:
                interp = "sensitivity-dominant degradation"
            elif d_ba > -0.03:
                interp = "minimal degradation"
            else:
                interp = "mixed degradation"

            print(f"  {model:20s} {style:12s}  "
                  f"{d_ba:+8.3f} {d_sens:+8.3f} {d_spec:+8.3f}  "
                  f"{interp:30s}")

    print()
    print("  Prediction (pre-registered): specificity-dominant degradation.")
    print("  If sensitivity-dominant or minimal: engages less-is-more hypothesis")
    print("  directly — note this in the writeup.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--split",   default="dev")
    args = ap.parse_args()

    results_dir = ROOT / args.results
    all_data = load_results(results_dir, args.split)

    if not all_data:
        print(f"No result files found in {results_dir} for split={args.split}.")
        print("Run eval/run.py first.")
        return

    # Deduplicate: if multiple files for same model/style/level, keep most recent
    from collections import defaultdict
    by_condition: dict[tuple, list] = defaultdict(list)
    for data in all_data:
        c = data.get("_condition", {})
        key = (c.get("model",""), c.get("style",""), c.get("level",""))
        by_condition[key].append(data)
    all_data = []
    for key, dlist in by_condition.items():
        # sort by timestamp in metrics label or filename, keep last
        dlist.sort(key=lambda d: d.get("metrics", {}).get("label", ""), reverse=True)
        all_data.append(dlist[0])

    rows = []
    for data in all_data:
        cond = data.get("_condition", {})
        model = cond.get("model", "?")
        style = cond.get("style", "?")
        level = cond.get("level", "?")
        m   = data["metrics"]
        row = calibration_row(data["results"], m)
        row.update({"model": model, "style": style, "level": level})
        rows.append(row)

    # sort: model, style, level order
    level_order = {"full": 0, "claims_only": 1, "claims_only*": 1, "output_only": 2}
    rows.sort(key=lambda r: (r["model"], r["style"],
                              level_order.get(r["level"], 9)))

    print_table(rows, f"CALIBRATION TABLE — {args.split} split")
    source_access_effect(rows)

    print("\nINTERPRETATION GUIDE")
    print("  H1 supported if: structured_full BA CI excludes holistic_full BA.")
    print("  Mechanism confirmed if: structured_full specificity >> holistic_full")
    print("    specificity, with sensitivity roughly matched.")
    print("  Source-access novel finding if: full→output_only degrades specificity")
    print("    more than sensitivity (predicted), OR reverses (more interesting).")


if __name__ == "__main__":
    main()