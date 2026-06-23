"""
analysis/resume_failed.py — re-run api_error records from a specific result file
and patch them back in.

Usage:
  python analysis/resume_failed.py <result_file.json>

Example:
  python analysis/resume_failed.py results/test_holistic_output_only_openai_gpt_oss_20b_20260622_001232.json

This completes the frozen test — it re-requests the same records with the same
prompt and seed, filling in the 4 null-content failures. It does not change any
other records or re-run anything that already has a valid prediction.
"""

from __future__ import annotations
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "monitors"))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "eval"))

import source_access as sa
from schema import load_jsonl, iter_split, example_pool


def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis/resume_failed.py <result_file.json>")
        sys.exit(1)

    result_path = Path(sys.argv[1])
    data = json.loads(result_path.read_text())
    results = data["results"]

    # Find records with no valid prediction
    failed = [r for r in results if r.get("pred_label") is None]
    print(f"Found {len(failed)} failed records in {result_path.name}")
    if not failed:
        print("Nothing to retry.")
        return

    # Load dataset to get full Record objects and example pool
    recs = load_jsonl(ROOT / "data" / "dataset.jsonl")
    pool = example_pool(recs)
    rec_by_id = {r.id: r for r in recs}

    # Get condition from first result
    r0 = results[0]
    model  = r0["model"]
    style  = r0["style"]
    level  = r0["source_level"]
    n_shots = r0.get("n_shots", 3)

    print(f"Condition: model={model} style={style} level={level} shots={n_shots}")
    print(f"Retrying {len(failed)} records...\n")

    fixed = 0
    for r in failed:
        rec = rec_by_id.get(r["id"])
        if rec is None:
            print(f"  {r['id']}: record not found in dataset, skipping")
            continue
        result = sa.predict(rec, pool, model, style, level, n_shots)
        if result["pred_label"] is not None:
            # Patch the result back in
            r.update(result)
            fixed += 1
            print(f"  {r['id']}: {result['pred_str']} (was api_error)")
        else:
            print(f"  {r['id']}: still failed — {result.get('raw_response','')[:60]}")

    # Save patched file
    # Recompute metrics
    from metrics import compute_metrics, bootstrap_ci
    kept = [r for r in results if r.get("pred_label") is not None]
    m  = compute_metrics(results, label=data["metrics"].get("label", ""))
    ci = bootstrap_ci(results, seed=42)
    data["results"] = results
    data["metrics"] = m
    data["bootstrap_ci"] = ci
    result_path.write_text(json.dumps(data, indent=2))
    print(f"\nPatched {fixed}/{len(failed)} records.")
    print(f"New n_parseable: {len(kept) + fixed}")
    print(f"Saved to {result_path}")


if __name__ == "__main__":
    main()