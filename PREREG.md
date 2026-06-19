# PREREG: Structured vs Holistic Monitoring for Detecting Strategic Omission

Frozen before the full v0 evaluation. Deviations from this document will be
reported explicitly in the writeup (Section 6.6).

## Question

Does structured, decomposed monitoring outperform holistic monitoring at
detecting strategically omitted or downplayed safety-relevant information, and
how does that advantage depend on (a) monitor capability and (b) adversarial
pressure from the policy model?

## Conditions and labels

Binary task. `faithful` = label 0 (target present). `omission` and
`adversarial_omission` = label 1 (target removed or softened). v0 covers
`faithful` and mechanically-generated `omission` only; `adversarial_omission`
arrives in Phase 3.

## Hypotheses (directional predictions committed now)

**H1.** For a fixed monitor, structured monitoring achieves higher balanced
accuracy than holistic monitoring at detecting omission on clean
(non-adversarial) data.

**H2.** The structured advantage is largest for the weakest monitor and shrinks
as monitor capability rises; for the strongest monitor it may vanish or invert.
Rationale committed in advance: in the predecessor SEC study, a structured
representation moved the weakest model from near-zero to its best score, while
the same structuring hurt a stronger model (an "addition blindness" failure).
The prediction is therefore a decreasing, possibly sign-changing, advantage in
monitor capability, not a constant gain.

**H3 (most decision-relevant).** Under `adversarial_omission`, the structured
advantage shrinks but persists while the policy is no more capable than the
monitor, and collapses (possibly inverts) once the policy is more capable than
the monitor. Rationale committed in advance: adversarial paraphrase-around-the-gap
is expected to defeat per-claim decomposition specifically, by making each claim
check pass locally while the safety-relevant content is globally absent.

A clean null on any hypothesis is a reportable result. The analysis will not be
re-pointed to chase a positive.

## Primary metric

**Balanced accuracy**, chosen because recall on one class can be gamed by a
decision-threshold shift. Reported alongside: Cohen's kappa, confusion matrices,
and per-condition breakdowns.

## Primary comparison

Structured vs holistic, evaluated **on the same examples**, compared with a
**paired bootstrap** (resample examples with replacement; recompute the balanced
accuracy difference structured minus holistic on each resample; report the 95%
percentile CI of that difference). The advantage is called real only if the CI
excludes zero. Independent per-method CIs are reported for context but are not
the test.

## Fairness control

Holistic and structured monitors receive **exactly the same inputs** (source plus
policy output). The structured monitor's only permitted advantage is its
decomposition prompt. Neither monitor is given the ground-truth claim list, the
target id, or any directional cue. Rendered prompts will be audited for leakage
before the run (Section 6.2).

## Baseline (mandatory floor and confound check)

At least one non-LLM baseline (claim-overlap via embedding similarity, plus a
length-delta feature). Two checks:
1. The LLM monitors must beat the baseline, or the representation rather than the
   model is doing the work.
2. The baseline must NOT trivially solve the task. If it does, the construction
   has a confound and v0 is rebuilt before any LLM run.

Length is already controlled at generation time: a length-only classifier on the
v0 dataset sits at chance (see the generator's length-balance report).

## Monitor capability ladder (v0)

Policy fixed (mechanical omission). Monitors span a capability range, for example
one small open model, one mid model, one frontier model. Exact model list is
recorded in `results/` at run time; the ladder, not any single model, is the unit
of analysis for H2.

## Splits

train / dev / test, stratified by domain, split at the matched-pair level so a
source document never straddles splits. **Test is frozen.** All prompt design and
iteration happen on train/dev. Test is evaluated once, at the end.

## Go / no-go

If structured shows a CI-separated advantage over holistic on clean data (H1),
proceed to Phase 2 scaling. If not, document the null and decide whether the
question is still worth scaling. The decision is made on dev, then confirmed once
on test.
