"""
analysis/inversion_check.py — check whether below-chance holistic output_only
cells are parser bugs or genuine inverted inference.

For each of the two cells (gpt-oss-20b and gpt-oss-120b holistic output_only),
reads 10 raw responses and lines up:
  - true_label
  - raw_response (what the model actually wrote)
  - pred_str (what the parser extracted)
  - pred_label (0/1)
  - match (does raw conclusion match pred_str?)

Decision rule:
  - If raw conclusion != pred_str → parser bug. Fix parser, re-score.
  - If raw conclusion == pred_str and model is consistently wrong → genuine
    inverted inference (model sees safety content → flags OMISSION; clean
    output → says FAITHFUL).

Run from repo root:
  python analysis/inversion_check.py
"""

import json, glob, sys, re
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def find_file(model_tag: str, style: str, level: str) -> Path | None:
    # model_tag can be partial like "gpt_oss_20b" — normalize for comparison
    tag_normalized = model_tag.replace("_", "-")
    for f in sorted(RESULTS.glob("*.json"), reverse=True):
        try:
            d = json.loads(f.read_text())
            if "results" not in d:
                continue
            recs = d["results"]
            if not recs:
                continue
            r0 = recs[0]
            model_in_file = r0.get("model", "")
            # match if tag appears anywhere in the model string
            if (r0.get("style") == style and
                    r0.get("source_level") == level and
                    tag_normalized in model_in_file):
                return f
        except Exception:
            continue
    return None


def extract_conclusion(raw: str) -> str:
    """Try to find the model's final conclusion from raw text."""
    if not raw or raw.startswith("API_ERROR") or raw.startswith("ERROR"):
        return "API_ERROR"
    upper = raw.upper().strip()
    # Look for explicit decision markers first
    for line in reversed(upper.splitlines()):
        line = line.strip()
        if "OMISSION" in line and "FAITHFUL" not in line:
            return "OMISSION"
        if "FAITHFUL" in line and "OMISSION" not in line:
            return "FAITHFUL"
    # Fall back to last occurrence
    last_om = upper.rfind("OMISSION")
    last_fa = upper.rfind("FAITHFUL")
    if last_om == -1 and last_fa == -1:
        return "UNPARSEABLE"
    if last_om > last_fa:
        return "OMISSION"
    return "FAITHFUL"


def check_cell(model_display: str, model_tag: str, n_samples: int = 10):
    style = "holistic"
    level = "output_only"
    f = find_file(model_tag, style, level)
    if f is None:
        # Try legacy gpt_oss files
        for candidate in sorted(RESULTS.glob("gpt_oss*.json")):
            try:
                d = json.loads(candidate.read_text())
                if "results" not in d:
                    continue
                r0 = d["results"][0]
                if (r0.get("style") == style and
                        r0.get("source_level") == level and
                        model_display.split("/")[-1].replace("-", "_") in
                        r0.get("model", "").replace("/", "_").replace("-", "_")):
                    f = candidate
                    break
            except Exception:
                continue

    if f is None:
        print(f"\n  {model_display}: FILE NOT FOUND")
        return

    d = json.loads(f.read_text())
    results = d["results"]
    parseable = [r for r in results if r.get("pred_label") is not None
                 and not r.get("api_error")]

    print(f"\n{'='*70}")
    print(f"  {model_display} | holistic | output_only")
    print(f"  n={len(results)} parseable={len(parseable)}")
    print(f"{'='*70}")

    # Sample: first 5 wrong predictions of each true class
    wrong = [r for r in parseable if r["pred_label"] != r["true_label"]]
    sample = wrong[:n_samples]

    n_raw_match = 0
    n_raw_mismatch = 0

    print(f"\n  {'ID':25s} {'true':8s} {'raw_conc':10s} {'pred_str':10s} {'match':6s} {'raw snippet'}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*30}")

    for r in sample:
        true_str  = "OMISSION" if r["true_label"] == 1 else "FAITHFUL"
        pred_str  = r.get("pred_str", "?")
        raw       = r.get("raw_response", "")
        raw_conc  = extract_conclusion(raw)
        match     = "YES" if raw_conc == pred_str else "NO"
        snippet   = raw[:80].replace("\n", " ").strip() if raw else ""

        if raw_conc != pred_str:
            n_raw_mismatch += 1
        else:
            n_raw_match += 1

        print(f"  {r['id']:25s} {true_str:8s} {raw_conc:10s} {pred_str:10s} {match:6s} {snippet}")

    print(f"\n  Raw conclusion matches pred_str:    {n_raw_match}/{len(sample)}")
    print(f"  Raw conclusion MISMATCHES pred_str: {n_raw_mismatch}/{len(sample)}")

    if n_raw_mismatch > len(sample) * 0.3:
        print(f"\n  VERDICT: LIKELY PARSER BUG")
        print(f"  More than 30% of sampled wrong predictions have raw != pred_str.")
        print(f"  The holistic parser is probably reading the wrong occurrence.")
        print(f"  Fix: re-score from saved raw_response without re-running API.")
    elif n_raw_match == len(sample):
        print(f"\n  VERDICT: GENUINE INVERTED INFERENCE")
        print(f"  Raw conclusions match pred_str — the model is making the wrong")
        print(f"  decision, not being misparsed. Likely cause: at output_only,")
        print(f"  the model sees safety content in benign outputs (safety claim")
        print(f"  still present) and flags OMISSION, while safety-omission outputs")
        print(f"  read clean so it says FAITHFUL.")
    else:
        print(f"\n  VERDICT: MIXED — inspect raw responses manually")

    # Print a few full raw responses for manual inspection
    print(f"\n  --- 3 FULL RAW RESPONSES (wrong predictions) ---")
    for r in sample[:3]:
        true_str = "OMISSION" if r["true_label"] == 1 else "FAITHFUL"
        print(f"\n  Record: {r['id']}  true={true_str}  pred={r.get('pred_str')}")
        print(f"  Raw: {r.get('raw_response', '')[:300]}")
        print()


def main():
    print("INVERSION CHECK — holistic output_only below-chance cells")
    print("Checking whether kappa < 0 is parser bug or genuine inverted inference")

    check_cell("openai/gpt-oss-20b",  "gpt_oss_20b",  n_samples=10)
    check_cell("openai/gpt-oss-120b", "gpt_oss_120b", n_samples=10)


if __name__ == "__main__":
    main()