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

---

## Phase 2 amendment (committed before the final test run)

This section extends the pre-registration to cover the Phase 2 design.
Written and committed before `--split test` is executed. Deviations will
be reported explicitly.

### Exact conditions in the final test run

Models: llama3:latest (weak), gpt-4o-mini (mid), gpt-4o (frontier).
Styles: holistic, structured.
Levels: full, claims_only_untagged, output_only.
Shots: 3. Split: test (n=120, 60 faithful / 60 omission, never touched).

`claims_only_untagged` is the canonical claims-only condition: an enumerated
list of claims with no safety labels, confirmed on dev to produce the same
near-ceiling performance as the tagged version, ruling out label leakage.

### Primary metric (unchanged)

Balanced accuracy. Reported with 95% bootstrap CIs and Cohen's kappa.
Paired bootstrap for all style comparisons on the same examples.

### Pre-registered comparisons and predicted directions

All predictions stated before the test run. CIs computed post-hoc.

**C1. Structured vs holistic at full source (H1, primary).**
Predicted direction: structured > holistic for mid and frontier models.
For Llama3 8B: null predicted (calibration swap observed on dev — both
styles converge at BA ~0.65 through opposite errors; no net advantage).
Decision rule: CI excludes zero = supported. CI includes zero = null.
Note: ~65% power at n=120 for the 0.125 dev effect. A null here is
uninformative, not a failure.

**C2. Output-only gap: structured vs holistic at output_only (headline).**
Predicted direction: structured > holistic, expected large effect (~0.25).
~95% power at n=120. This is the primary novel claim. A null here would
be a genuine failure to replicate.
Decision rule: CI excludes zero = supported.

**C3. Per-style source-access degradation: full vs output_only.**
Holistic: predicted specificity-dominant degradation (loses reference,
over-flags). Predicted direction: full > output_only, large effect.
Structured: predicted sensitivity-dominant degradation (loses ability to
detect absence). Predicted direction: full > output_only, smaller effect
than holistic. The asymmetry between styles is the novel mechanistic claim.

**C4. Claims-only vs full.**
Predicted: claims_only_untagged near-ceiling for mid and frontier models,
confirming that enumerated reference access is the bottleneck. For Llama3
8B: predicted claims_only_untagged < full (dev showed sensitivity collapse
without the prose framing).

**C5. Capability ladder (H2).**
Predicted: structured advantage over holistic increases with capability.
At Llama3 8B: null or reversed. At GPT-4o-mini: positive CI-separated.
At GPT-4o: positive, at least as large as GPT-4o-mini.
A monotone increasing advantage across the ladder confirms H2.

### What gets reported regardless

Every cell in the 3×2×3 grid gets reported, including any that fail to
replicate dev findings. No cells are dropped, re-run, or re-prompted after
seeing test numbers. The test run is executed once.

### Power note

The 0.250 output-only effect (C2) is powered at ~95% at n=120.
The 0.125 full-source effect (C1) is powered at ~65% at n=120.
A null on C1 at test is therefore consistent with a real effect and will
be reported as such, not as a contradiction of dev findings.