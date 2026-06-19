"""
eval/run.py — evaluation loop for the omission-monitoring project.

Runs holistic and structured monitors across the monitor capability ladder
on the frozen dev split (for iteration) or test split (once, at the end).

Usage examples:

  # dry run on dev — prints prompts, no API calls
  python eval/run.py --split dev --model llama3:latest --dry_run

  # v0 run on dev with one model
  python eval/run.py --split dev --model llama3:latest --shots 3

  # run both monitors on dev for two models
  python eval/run.py --split dev --model llama3:latest gpt-4o-mini --shots 3

  # run baseline only
  python eval/run.py --split dev --baseline_only

  # FINAL evaluation on test (do this ONCE at the very end)
  python eval/run.py --split test --model llama3:latest gpt-4o-mini --shots 3

Guardrails enforced here:
  - Default split is dev. Test requires --split test explicitly plus
    --i_have_read_prereg (a flag that forces you to acknowledge the pre-reg).
  - Results are saved to results/<split>_<monitor>_<model>_<timestamp>.json
    so nothing is overwritten.
  - The baseline confound check runs first; the script warns (but does not
    halt by default) if baseline BA >= 0.65. Pass --halt_on_confound to halt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# allow running from repo root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "monitors"))
sys.path.insert(0, str(ROOT / "eval"))

from schema import load_jsonl, iter_split, example_pool
import holistic as hol
import structured as stc
from baseline import run_baseline
from metrics import (
    compute_metrics, bootstrap_ci, paired_bootstrap_comparison, print_metrics
)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  saved → {path}")


def run_monitor(monitor_name: str, records: list, examples: list,
                model: str, n_shots: int, dry_run: bool) -> list[dict]:
    """Run holistic or structured monitor over all records."""
    predict_fn = hol.predict if monitor_name == "holistic" else stc.predict
    build_fn   = hol.build_prompt if monitor_name == "holistic" else stc.build_prompt

    results = []
    n = len(records)
    print(f"\n  Running {monitor_name} | model={model} | shots={n_shots} | n={n}")

    for i, rec in enumerate(records):
        if dry_run and i == 0:
            prompt = build_fn(rec, examples, n_shots, model)
            print(f"\n  --- SAMPLE PROMPT ({monitor_name}) ---")
            print(prompt[:1200] + ("\n  [truncated]" if len(prompt) > 1200 else ""))
            print("  --- END PROMPT ---\n")
            continue

        result = predict_fn(rec, examples, model, n_shots)
        results.append(result)
        label_str = "OMISSION" if result["true_label"] == 1 else "FAITHFUL"
        pred_str  = result["pred_str"]
        match = "✓" if result["pred_label"] == result["true_label"] else "✗"
        print(f"    {match} [{i+1:03d}/{n}] {rec.id[:40]:40s} "
              f"true={label_str:8s} pred={pred_str}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    default="data/dataset.jsonl")
    ap.add_argument("--results", default="results")
    ap.add_argument("--split",   default="dev", choices=["dev", "test"])
    ap.add_argument("--model",   nargs="+", default=["llama3:latest"],
                    help="One or more models. gpt-* routes to OpenAI, else Ollama.")
    ap.add_argument("--shots",   type=int, default=3)
    ap.add_argument("--monitors", nargs="+", default=["holistic", "structured"],
                    choices=["holistic", "structured"])
    ap.add_argument("--baseline_only", action="store_true")
    ap.add_argument("--dry_run",       action="store_true",
                    help="Print one sample prompt per monitor, no API calls.")
    ap.add_argument("--halt_on_confound", action="store_true",
                    help="Halt if baseline BA >= confound threshold.")
    ap.add_argument("--i_have_read_prereg", action="store_true",
                    help="Required when --split test. Confirms you've read PREREG.md.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ── test split gate ────────────────────────────────────────────────────────
    if args.split == "test" and not args.i_have_read_prereg:
        print("ERROR: Running on test requires --i_have_read_prereg.")
        print("Read PREREG.md, confirm this is the final run, then re-run with the flag.")
        sys.exit(1)

    # ── load data ──────────────────────────────────────────────────────────────
    data_path = ROOT / args.data
    print(f"Loading data from {data_path}")
    all_records = load_jsonl(data_path)

    train = iter_split(all_records, "train")
    dev   = iter_split(all_records, "dev")
    test  = iter_split(all_records, "test")
    eval_records = dev if args.split == "dev" else test
    examples = example_pool(all_records)   # train + dev only, never test

    print(f"Split: {args.split} ({len(eval_records)} records)")
    print(f"Example pool (train+dev): {len(examples)} records")

    results_dir = ROOT / args.results
    ts = timestamp()

    # ── baseline (always runs first) ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Non-LLM baseline (confound gate + floor check)")
    print("=" * 60)
    baseline_result = run_baseline(train, dev, eval_records)
    save(baseline_result, results_dir / f"{args.split}_baseline_{ts}.json")

    if baseline_result["confound_alert"] and args.halt_on_confound:
        print("\nHalting due to confound alert (--halt_on_confound set).")
        sys.exit(1)

    if args.baseline_only:
        return

    # ── LLM monitors ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: LLM monitors")
    print("=" * 60)

    all_monitor_results: dict[str, dict[str, list[dict]]] = {}
    # all_monitor_results[model][monitor_name] = list of result dicts

    for model in args.model:
        all_monitor_results[model] = {}
        for monitor_name in args.monitors:
            print(f"\n{'─'*60}")
            print(f"  model={model}  monitor={monitor_name}")
            print(f"{'─'*60}")

            results = run_monitor(
                monitor_name, eval_records, examples,
                model, args.shots, args.dry_run
            )

            if args.dry_run:
                continue

            # metrics
            tag = f"{args.split}_{monitor_name}_{model.replace(':','_').replace('-','_')}_{ts}"
            m   = compute_metrics(results, label=tag)
            ci  = bootstrap_ci(results, seed=args.seed)
            print_metrics(m)
            print(f"    95% CI (BA): [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

            save({"results": results, "metrics": m, "bootstrap_ci": ci},
                 results_dir / f"{tag}.json")

            all_monitor_results[model][monitor_name] = results

    if args.dry_run:
        return

    # ── paired bootstrap comparison (pre-registered primary test) ─────────────
    print("\n" + "=" * 60)
    print("STEP 3: Paired bootstrap comparison (pre-registered)")
    print("=" * 60)

    for model in args.model:
        mon_res = all_monitor_results.get(model, {})
        if "holistic" in mon_res and "structured" in mon_res:
            comp = paired_bootstrap_comparison(
                mon_res["structured"], mon_res["holistic"], seed=args.seed
            )
            save(comp, results_dir /
                 f"{args.split}_paired_comparison_{model.replace(':','_').replace('-','_')}_{ts}.json")
        else:
            print(f"  [{model}] Skipping paired comparison — need both monitors.")

    # ── go / no-go summary (dev only) ─────────────────────────────────────────
    if args.split == "dev":
        print("\n" + "=" * 60)
        print("GO / NO-GO SUMMARY (dev)")
        print("=" * 60)
        print("  Review the paired comparison verdicts above.")
        print("  GO criteria (from PREREG): CI of (BA_structured - BA_holistic) excludes zero.")
        print("  If GO: proceed to Phase 2 scaling.")
        print("  If NO-GO: document the null and decide whether to continue.")
        print("  When ready for the final test run:")
        print("    python eval/run.py --split test --i_have_read_prereg [same args]")
        print("=" * 60)


if __name__ == "__main__":
    main()
