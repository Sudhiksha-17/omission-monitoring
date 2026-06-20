"""
eval/run.py — Phase 2 evaluation loop.

Runs the 2 × 3 design: two prompt styles (holistic, structured) × three
source-access levels (full, claims_only, output_only), across the capability
ladder (Llama3 8B local + GPT-4o-mini), on the frozen dev split for iteration
and test once at the end.

GPT-4o is deferred to the final test run only (--defer_gpt4o flag, default on).

Usage:

  # dev run, Llama only (cheap, fast)
  python eval/run.py --split dev --model llama3:latest --shots 3

  # dev run, both ladder models
  python eval/run.py --split dev --model llama3:latest gpt-4o-mini --shots 3

  # run only specific styles/levels
  python eval/run.py --split dev --model llama3:latest --styles holistic structured --levels full output_only

  # dry run: print one sample prompt per condition, no API calls
  python eval/run.py --split dev --model llama3:latest --dry_run

  # FINAL test run (once, at the end, requires flag)
  python eval/run.py --split test --model llama3:latest gpt-4o-mini gpt-4o --i_have_read_prereg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "monitors"))
sys.path.insert(0, str(ROOT / "eval"))

from schema import load_jsonl, iter_split, example_pool
import source_access as sa
from baseline import run_baseline
from metrics import (
    compute_metrics, bootstrap_ci, paired_bootstrap_comparison, print_metrics
)

ALL_STYLES = list(sa.PROMPT_STYLES)
ALL_LEVELS = list(sa.SOURCE_LEVELS)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  saved → {path}")


def run_condition(style: str, level: str, records: list, examples: list,
                  model: str, n_shots: int, dry_run: bool) -> list[dict]:
    tag = f"{style}_{level}"
    n = len(records)
    print(f"\n  Running {tag:25s} | model={model} | shots={n_shots} | n={n}")

    if dry_run:
        sample_prompt = sa.build_prompt(records[0], examples, style, level, model, n_shots)
        print(f"\n  --- SAMPLE PROMPT ({tag}) ---")
        print(sample_prompt[:1400] + ("\n  [truncated]" if len(sample_prompt) > 1400 else ""))
        print("  --- END PROMPT ---\n")
        return []

    results = []
    for i, rec in enumerate(records):
        result = sa.predict(rec, examples, model, style, level, n_shots)
        results.append(result)
        label_str = "OMISSION" if result["true_label"] == 1 else "FAITHFUL"
        match = "✓" if result["pred_label"] == result["true_label"] else "✗"
        print(f"    {match} [{i+1:03d}/{n}] {rec.id[:38]:38s} "
              f"true={label_str:8s} pred={result['pred_str']}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    default="data/dataset.jsonl")
    ap.add_argument("--results", default="results")
    ap.add_argument("--split",   default="dev", choices=["dev", "test"])
    ap.add_argument("--model",   nargs="+", default=["llama3:latest"])
    ap.add_argument("--shots",   type=int, default=3)
    ap.add_argument("--styles",  nargs="+", default=ALL_STYLES,
                    choices=ALL_STYLES)
    ap.add_argument("--levels",  nargs="+", default=ALL_LEVELS,
                    choices=ALL_LEVELS)
    ap.add_argument("--baseline_only",    action="store_true")
    ap.add_argument("--dry_run",          action="store_true")
    ap.add_argument("--halt_on_confound", action="store_true")
    ap.add_argument("--i_have_read_prereg", action="store_true",
                    help="Required for --split test.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ── test gate ──────────────────────────────────────────────────────────────
    if args.split == "test" and not args.i_have_read_prereg:
        print("ERROR: --split test requires --i_have_read_prereg.")
        print("Read PREREG.md and PHASE2.md, confirm this is the final run.")
        sys.exit(1)

    # ── load ───────────────────────────────────────────────────────────────────
    data_path = ROOT / args.data
    print(f"Loading data from {data_path}")
    all_records = load_jsonl(data_path)

    train = iter_split(all_records, "train")
    dev   = iter_split(all_records, "dev")
    test  = iter_split(all_records, "test")
    eval_records = dev if args.split == "dev" else test
    pool  = example_pool(all_records)

    print(f"Split: {args.split} ({len(eval_records)} records) | "
          f"example pool: {len(pool)}")

    results_dir = ROOT / args.results
    ts = timestamp()

    # ── baseline ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Non-LLM baseline (confound gate + floor check)")
    print("=" * 60)
    bl = run_baseline(train, dev, eval_records)
    save(bl, results_dir / f"{args.split}_baseline_{ts}.json")
    if bl["confound_alert"] and args.halt_on_confound:
        print("\nHalting: confound alert + --halt_on_confound set.")
        sys.exit(1)
    if args.baseline_only:
        return

    # ── 2 × 3 monitor grid ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: 2 × 3 monitor grid (style × source_access)")
    print(f"  styles: {args.styles}")
    print(f"  levels: {args.levels}")
    print("=" * 60)

    # all_results[model][style][level] = list[dict]
    all_results: dict[str, dict[str, dict[str, list[dict]]]] = {}

    for model in args.model:
        all_results[model] = {s: {} for s in args.styles}
        for style in args.styles:
            for level in args.levels:
                print(f"\n{'─'*60}")
                print(f"  model={model}  style={style}  level={level}")
                print(f"{'─'*60}")
                results = run_condition(
                    style, level, eval_records, pool,
                    model, args.shots, args.dry_run
                )
                if args.dry_run or not results:
                    continue

                tag = (f"{args.split}_{style}_{level}_"
                       f"{model.replace(':','_').replace('-','_')}_{ts}")
                m  = compute_metrics(results, label=tag)
                ci = bootstrap_ci(results, seed=args.seed)
                print_metrics(m)
                print(f"    95% CI (BA): [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")
                save({"results": results, "metrics": m, "bootstrap_ci": ci},
                     results_dir / f"{tag}.json")
                all_results[model][style][level] = results

    if args.dry_run:
        return

    # ── primary paired comparisons ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Paired bootstrap comparisons")
    print("=" * 60)

    for model in args.model:
        mr = all_results.get(model, {})

        # H1: structured_full vs holistic_full (pre-registered)
        s_full = mr.get("structured", {}).get("full", [])
        h_full = mr.get("holistic",   {}).get("full", [])
        if s_full and h_full:
            print(f"\n  [{model}] H1: structured_full vs holistic_full")
            comp = paired_bootstrap_comparison(s_full, h_full, seed=args.seed)
            save(comp, results_dir /
                 f"{args.split}_H1_paired_{model.replace(':','_').replace('-','_')}_{ts}.json")

        # Source-access effect within holistic
        for lvl_a, lvl_b in [("full", "claims_only"),
                               ("full", "output_only"),
                               ("claims_only", "output_only")]:
            ra = mr.get("holistic", {}).get(lvl_a, [])
            rb = mr.get("holistic", {}).get(lvl_b, [])
            if ra and rb:
                print(f"\n  [{model}] holistic: {lvl_a} vs {lvl_b}")
                paired_bootstrap_comparison(ra, rb, seed=args.seed)

        # Source-access effect within structured
        for lvl_a, lvl_b in [("full", "claims_only"),
                               ("full", "output_only")]:
            ra = mr.get("structured", {}).get(lvl_a, [])
            rb = mr.get("structured", {}).get(lvl_b, [])
            if ra and rb:
                print(f"\n  [{model}] structured: {lvl_a} vs {lvl_b}")
                paired_bootstrap_comparison(ra, rb, seed=args.seed)

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Summary table (BA by model × style × level)")
    print("=" * 60)
    print(f"  {'model':20s} {'style':12s} {'level':14s} {'BA':>7s} {'kappa':>7s}")
    print(f"  {'-'*20} {'-'*12} {'-'*14} {'-'*7} {'-'*7}")
    for model in args.model:
        for style in args.styles:
            for level in args.levels:
                res = all_results.get(model, {}).get(style, {}).get(level)
                if res:
                    m = compute_metrics(res)
                    print(f"  {model:20s} {style:12s} {level:14s} "
                          f"{m['balanced_accuracy']:7.4f} {m['kappa']:7.4f}")

    if args.split == "dev":
        print("\n" + "=" * 60)
        print("GO / NO-GO (dev)")
        print("=" * 60)
        print("  H1 (structured_full > holistic_full): see paired comparison above.")
        print("  Source-access: inspect holistic full→output_only degradation.")
        print("  Key question: is specificity destroyed more than sensitivity?")
        print("  Run analysis/calibration.py for the breakdown.")
        print("  Final test run when ready:")
        print("    python eval/run.py --split test --i_have_read_prereg [args]")
        print("=" * 60)


if __name__ == "__main__":
    main()
