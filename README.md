# Omission Monitoring

Can an overseer detect when a more capable system quietly omits or downplays
important information, and what does the overseer need in order to catch it?

This repo moves that question into a clean, controlled setting: a trusted monitor
watches an untrusted policy whose subversion is **omission** (dropping or
softening a safety-relevant claim), not code sabotage. The core comparison is
**structured / decomposed monitoring vs holistic monitoring** on identical
inputs, across a monitor capability ladder, with and without adversarial pressure.

See `PREREG.md` for the frozen hypotheses, primary metric, and exact comparison.

## Why this is not already answered

The AI-control monitoring literature centers on code sabotage and backdoors
(Greenblatt et al., AI Control, 2024) and on what the monitor observes (CoT vs
action; Arnav et al., CoT Red-Handed, 2025). Recent decomposition monitors
(TraceGuard; Factor(T,U)) show structure helps in agentic CoT settings. None
isolates omission of a discrete, enumerable safety-relevant claim with exact
ground truth, and none cleanly separates the prompt-decomposition variable
(one monitor, same inputs, structured vs holistic prompt only) from monitor
capability and adversarial pressure. The contribution is that clean separation,
plus a single decision-relevant number: the policy-to-monitor capability gap at
which reliable omission detection breaks down.

## Layout

```
data/
  schema.py        # record schema, validation, strict split handling
  generate.py      # v0 synthetic generator (matched, length-controlled pairs)
  dataset.jsonl    # generated dataset (regenerate with generate.py)
  splits.json      # split summary
monitors/
  holistic.py      # single-prompt binary omission judgment        [next]
  structured.py    # enumerate -> check each claim -> aggregate     [next]
  baseline.py      # non-LLM floor (claim-overlap + length-delta)   [next]
eval/
  run.py           # eval loop across monitor ladder + conditions   [next]
  metrics.py       # balanced acc, kappa, bootstrap + paired bootstrap [next]
results/
analysis/
  figures.py                                                        [next]
PREREG.md          # frozen hypotheses, metric, comparison
README.md
```

## Status

- [x] Repo skeleton, schema, strict split handling
- [x] PREREG frozen
- [x] v0 generator: 40 matched faithful/omission pairs, length-controlled, exact labels
- [ ] holistic + structured monitors, non-LLM baseline
- [ ] eval loop + metrics (balanced acc, kappa, paired bootstrap CIs)
- [ ] v0 run across the monitor ladder, go / no-go call

## Reproduce v0 data

```bash
cd data
python generate.py --n 40 --out . --seed 7
```

This writes `dataset.jsonl` and `splits.json` and prints a length-balance report.
The report must show `"balanced": true` (a length-only classifier at chance)
before any monitor is run; that is the Section 6.3 confound gate.

## Provenance

The evaluation loop and Cohen's kappa scoring are adapted from the predecessor
SEC study's harness (`call_llm` model routing, the prompt-routing-by-condition
pattern, and the YES/NO elicitation scaffold). Finance-specific machinery
(diff signal tags, COVID heuristics, tiebreaker and quality whitelists, the
3-class scheme) was deliberately dropped. The k-fold CV from that harness is
replaced by a frozen train/dev/test split, and silent defaulting of unparseable
model output is removed, both to close leakage paths flagged in that project.
