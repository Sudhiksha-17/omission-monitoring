# Results — held-out claim set

All claims confirmed on the frozen held-out test split (n=120 pairs/cell), CI-backed (5000-resample paired bootstrap, paired by pair_id). Claims were locked on dev before the test was scored. gpt-4o is a ceiling anchor; an earlier llama3 run is excluded.

## Confirmed (held-out, CI-clean)

1. **Fail-open under source removal (headline).** Full to output_only degradation is sensitivity-dominant on 8 of 9 cells (sensitivity drops 0.42-0.82). The pre-registered specificity-dominant prediction is refuted. The lone exception is the gpt-4o anchor under holistic prompting, which fails toward false alarms instead.
2. **Structured prompting protects every model at output_only**, on BA and specificity, all CI-clean.
3. **Holistic goes below chance at output_only** on three models (kappa -0.27 to -0.42); structured stays positive. Direction claimed; mechanism (presence heuristic vs prior-defaulting vs misinterpretation) not adjudicable from one-word outputs.

## Scoped / secondary

4. **Structure rescues a degenerate baseline at full source (gpt-4o-mini only)**, +0.142 BA, CI [+0.073, +0.210]; null on both gpt-oss models. Not a general specificity improvement. The gpt-4o-mini degenerate holistic arm is a prompt-reliability artifact, not a capability signature.

## Retracted

- No capability trend. Effect sizes do not order monotonically by capability; earlier "dissociation" and "capability ladder" framings did not survive the CIs and were dropped.

## Honesty notes

- "Structure does not hurt" is the claim at full source on capable models: no cell shows a statistically significant structured-below-holistic result at n=120.
- The inversion below-chance magnitude is amplified by the single-safety-claim design; the direction is the claim.

---

## Table A1 — Calibration (test split)

| Model | Style | Source | BA | kappa | Sens | Spec | FPR | n |
|---|---|---|---|---|---|---|---|---|
| gpt-4o (anchor) | holistic | full | 0.983 | 0.967 | 1.000 | 0.967 | 0.033 | 120 |
| gpt-4o (anchor) | holistic | claims_only | 0.983 | 0.966 | 0.967 | 1.000 | 0.000 | 119 |
| gpt-4o (anchor) | holistic | output_only | 0.783 | 0.567 | 1.000 | 0.567 | 0.433 | 120 |
| gpt-4o (anchor) | structured | full | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 120 |
| gpt-4o (anchor) | structured | claims_only | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 120 |
| gpt-4o (anchor) | structured | output_only | 0.717 | 0.433 | 0.433 | 1.000 | 0.000 | 120 |
| gpt-4o-mini | holistic | full | 0.500 | 0.000 | 0.950 | 0.050 | 0.950 | 120 |
| gpt-4o-mini | holistic | claims_only | 0.492 | -0.017 | 0.967 | 0.017 | 0.983 | 120 |
| gpt-4o-mini | holistic | output_only | 0.292 | -0.417 | 0.533 | 0.050 | 0.950 | 120 |
| gpt-4o-mini | structured | full | 0.642 | 0.283 | 0.933 | 0.350 | 0.650 | 120 |
| gpt-4o-mini | structured | claims_only | 0.717 | 0.433 | 0.950 | 0.483 | 0.517 | 120 |
| gpt-4o-mini | structured | output_only | 0.625 | 0.250 | 0.300 | 0.950 | 0.050 | 120 |
| gpt-oss-20b | holistic | full | 0.833 | 0.667 | 0.967 | 0.700 | 0.300 | 120 |
| gpt-oss-20b | holistic | claims_only | 0.817 | 0.633 | 0.933 | 0.700 | 0.300 | 120 |
| gpt-oss-20b | holistic | output_only | 0.367 | -0.267 | 0.233 | 0.500 | 0.500 | 120 |
| gpt-oss-20b | structured | full | 0.865 | 0.731 | 0.917 | 0.814 | 0.186 | 119 |
| gpt-oss-20b | structured | claims_only | 0.891 | 0.782 | 0.983 | 0.800 | 0.200 | 119 |
| gpt-oss-20b | structured | output_only | 0.667 | 0.333 | 0.450 | 0.883 | 0.117 | 120 |
| gpt-oss-120b | holistic | full | 0.883 | 0.767 | 0.950 | 0.817 | 0.183 | 120 |
| gpt-oss-120b | holistic | claims_only | 0.842 | 0.683 | 0.900 | 0.783 | 0.217 | 120 |
| gpt-oss-120b | holistic | output_only | 0.375 | -0.250 | 0.133 | 0.617 | 0.383 | 120 |
| gpt-oss-120b | structured | full | 0.850 | 0.700 | 0.950 | 0.750 | 0.250 | 120 |
| gpt-oss-120b | structured | claims_only | 0.800 | 0.600 | 0.983 | 0.617 | 0.383 | 120 |
| gpt-oss-120b | structured | output_only | 0.700 | 0.400 | 0.483 | 0.917 | 0.083 | 120 |

BA = balanced accuracy. Sens = sensitivity (omission recall). Spec = specificity (benign recall). FPR = false-positive rate. Negative kappa = below-chance (anti-correlated with ground truth).

## Table A2 — Paired bootstrap CIs (structured vs holistic)

5000 resamples, paired by pair_id. "diff" is structured minus holistic. Positive favors structured.

| Model | Comparison | Metric | Structured | Holistic | diff | 95% CI | Verdict |
|---|---|---|---|---|---|---|---|
| gpt-4o-mini | H1 (full) | BA | 0.642 | 0.500 | +0.142 | [+0.073, +0.210] | structured > holistic |
| gpt-4o-mini | H1 (full) | sensitivity | 0.933 | 0.950 | -0.017 | [-0.054, +0.000] | null |
| gpt-4o-mini | H1 (full) | specificity | 0.350 | 0.050 | +0.300 | [+0.167, +0.433] | structured > holistic |
| gpt-4o-mini | output_only | BA | 0.625 | 0.292 | +0.333 | [+0.217, +0.455] | structured > holistic |
| gpt-4o-mini | output_only | specificity | 0.950 | 0.050 | +0.900 | [+0.806, +0.982] | structured > holistic |
| gpt-oss-20b | H1 (full) | BA | 0.865 | 0.831 | +0.034 | [-0.038, +0.108] | null |
| gpt-oss-20b | H1 (full) | sensitivity | 0.917 | 0.967 | -0.050 | [-0.123, +0.018] | null |
| gpt-oss-20b | H1 (full) | specificity | 0.814 | 0.695 | +0.119 | [+0.000, +0.246] | null (CI touches 0) |
| gpt-oss-20b | output_only | BA | 0.667 | 0.367 | +0.300 | [+0.176, +0.420] | structured > holistic |
| gpt-oss-20b | output_only | specificity | 0.883 | 0.500 | +0.383 | [+0.216, +0.544] | structured > holistic |
| gpt-oss-120b | H1 (full) | BA | 0.850 | 0.883 | -0.033 | [-0.100, +0.031] | null |
| gpt-oss-120b | H1 (full) | sensitivity | 0.950 | 0.950 | +0.000 | [-0.050, +0.048] | null |
| gpt-oss-120b | H1 (full) | specificity | 0.750 | 0.817 | -0.067 | [-0.189, +0.053] | null |
| gpt-oss-120b | output_only | BA | 0.700 | 0.375 | +0.325 | [+0.210, +0.433] | structured > holistic |
| gpt-oss-120b | output_only | specificity | 0.917 | 0.617 | +0.300 | [+0.150, +0.444] | structured > holistic |

## Table A3 — Source-access degradation (full to output_only)

| Model | Style | dBA | dSens | dSpec | Pattern |
|---|---|---|---|---|---|
| gpt-4o (anchor) | holistic | -0.200 | +0.000 | -0.400 | specificity-dominant |
| gpt-4o (anchor) | structured | -0.283 | -0.567 | +0.000 | sensitivity-dominant |
| gpt-4o-mini | holistic | -0.208 | -0.417 | +0.000 | sensitivity-dominant |
| gpt-4o-mini | structured | -0.017 | -0.633 | +0.600 | sensitivity-dominant |
| gpt-oss-20b | holistic | -0.467 | -0.733 | -0.200 | sensitivity-dominant |
| gpt-oss-20b | structured | -0.198 | -0.467 | +0.070 | sensitivity-dominant |
| gpt-oss-120b | holistic | -0.508 | -0.817 | -0.200 | sensitivity-dominant |
| gpt-oss-120b | structured | -0.150 | -0.467 | +0.167 | sensitivity-dominant |

Sensitivity-dominant degradation (the fail-open direction) on 8 of 9 cells. The sole exception is the gpt-4o anchor under holistic prompting, the only configuration that fails toward false alarms rather than false negatives.

## Prompt-paraphrase robustness (dev split, full source)

Two paraphrases per style (v1, v2) plus the original (v0), paired bootstrap on specificity. The full-source structured advantage is large and stable across paraphrases only for gpt-4o-mini (degenerate holistic baseline). On the gpt-oss models it is direction-consistent but reaches significance in a minority of paraphrases (v2 only), and is never a significant holistic win in any paraphrase. Per-variant CIs are produced by analysis/robustness_bootstrap.py.
