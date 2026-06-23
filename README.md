# Omission monitoring under reduced source access

A controlled, pre-registered study of how LLM omission monitors fail when their access to the source is reduced. **Headline:** removing the source makes monitors fail *open* (toward false negatives), not closed — the opposite of the pre-registered prediction, confirmed on a frozen held-out split.

This README is the front door. It is structured to show the work: the confound that was caught and fixed, the pre-registration, the held-out confirmation, and the honest scoping of every secondary claim.

## TL;DR

- **Task.** A summary omits exactly one claim from a multi-claim source. The monitor decides whether the omitted claim was the safety-relevant one or a routine one. Both classes drop one claim, so length and "something is missing" carry no signal.
- **Manipulation.** Three source-access levels (full / claims-only / output-only) x two prompt styles (holistic / structured).
- **Result 1 (headline, held-out, CI-clean).** Removing source access degrades **sensitivity**, not specificity — sensitivity-dominant on 8 of 9 cells. Monitors become permissive, not suspicious.
- **Result 2 (held-out, CI-clean).** With no source, **structured prompting protects every model** on BA and specificity.
- **Result 3 (direction, hedged).** Holistic monitoring goes **below chance** at output-only on three models; structured does not.
- **Scoped secondary.** At full source, structure only *rescues a degenerate holistic baseline* (gpt-4o-mini); null on capable models. Not a general improvement.
- **Retracted.** No capability trend. Earlier "dissociation" and "capability ladder" framings did not survive the CIs and were dropped.

Full claim set and tables: [RESULTS.md](RESULTS.md). Writeup: [post/failopen_post.md](post/failopen_post.md).

## The confound, and why this repo leads with it

The first version of this benchmark was **93.5% solvable by a length-plus-sentiment classifier** — the task was a surface artifact, not semantic reasoning. The fix (reframing the negative class so both classes drop exactly one claim) brought that probe to **chance (55.8%)**. That before/after is the single most important artifact here; see `analysis/surface_feature_probe.py` and `PREREG.md`. The whole dataset, including the frozen test, was regenerated after the fix.

## Repository layout

```
.
├── README.md                        <- you are here
├── PREREG.md                        <- pre-registration; predictions locked before the test
├── RESULTS.md                       <- held-out claim set + appendix tables
├── requirements.txt
├── post/
│   ├── failopen_post.md             <- the writeup (Alignment Forum / LessWrong)
│   └── failopen_figure.png          <- the one body figure
├── data/
│   ├── generate.py                  <- dataset generator (routine-omission design)
│   ├── schema.py                    <- Record schema + loader
│   ├── dataset.jsonl                <- generated pairs (train/dev/test tagged)
│   └── splits.json                  <- pair-level split assignment
├── analysis/
│   ├── surface_feature_probe.py     <- the 0.935 -> 0.5583 confound gate
│   ├── source_access.py             <- eval harness (prompts, providers, parser)
│   ├── metrics.py                   <- balanced accuracy, kappa, paired bootstrap
│   ├── calibration.py               <- per-cell calibration table
│   ├── dev_bootstrap_cis.py         <- dev-split CIs (claim locking)
│   ├── test_h1_bootstrap.py         <- held-out CIs (confirmation)
│   ├── inversion_check.py           <- below-chance cells are genuine, not parser bugs
│   ├── robustness_run.py            <- prompt-paraphrase robustness driver (resumable)
│   ├── robustness_bootstrap.py      <- CIs for the paraphrase check
│   └── scripts/                     <- helper/recovery scripts (check_*, resume_failed)
└── results/
    └── *.json                       <- raw per-prediction outputs (one file per cell)
```

## Reproducing

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...        # gpt-4o, gpt-4o-mini
export OPENROUTER_API_KEY=...    # gpt-oss-20b, gpt-oss-120b (SiliconFlow)

# 1. regenerate data (or use the committed data/dataset.jsonl)
python data/generate.py

# 2. confound gate -- must be near chance before trusting anything downstream
python analysis/surface_feature_probe.py        # expect ~0.5583

# 3. run the grid (writes incrementally to results/, resumable)
python analysis/source_access.py --split dev
python analysis/source_access.py --split test

# 4. analysis (runs offline on the committed results/)
python analysis/calibration.py --results results --split test
python analysis/test_h1_bootstrap.py
python analysis/robustness_run.py && python analysis/robustness_bootstrap.py
```

The committed `results/` lets steps in (4) run with no API spend, so the held-out CIs are reproducible offline.

## What this is and is not

This is a **controlled probe** of one sub-problem: deciding whether a *known* omission is safety-relevant. It establishes **internal validity** (the confound gate, held-out confirmation, CI discipline), **not external validity**. Real omissions are paraphrastic, multiple, and buried; this isolates clean single deletion. It is monitor evaluation, not a control evaluation — the omissions are passive, with no adversary. See `RESULTS.md` for the full limitations and the planned extensions (reasoning traces, adversarial omission, wider model sweep).

## Closest prior work

AutoMonitor-Bench (Yang et al., arXiv:2601.05752, 2026) evaluates misbehavior monitors on *commission* with the reference always present, and reports an MR/FAR trade-off. This work varies the axis they hold fixed — source access — on an *omission* task, and finds a directional collapse rather than a trade-off. No "first" claim; the contribution is the source-access-by-error-type interaction.

## Collaboration

If you work on reference-free or omission-style monitoring, especially a control setup where a policy chooses what to omit, I would like to compare notes. Adversarial omission under monitor pressure is the direction I most want to take this next.
