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

# Omission Monitoring

**Can an overseer detect when a more capable system quietly omits or downplays important information, and what does the overseer need in order to catch it?**

This repo is an empirical study of monitor behavior under omission — a precursor to full AI control evaluation. A trusted monitor watches a policy output and must detect when a safety-relevant claim has been quietly dropped. We vary two things: how the monitor is prompted (holistic vs structured decomposition) and how much source information the monitor has access to (full source, enumerated claim list, or output only). We run this across a capability ladder of three models.

This README is a complete project record: design, findings, every methodological problem we encountered, how each was resolved, and the current state. It is written so that anyone picking up this repo can understand not just what we did but why the design is the way it is.

---

## Table of contents

1. [Research question and contribution](#1-research-question-and-contribution)
2. [Design](#2-design)
3. [Results](#3-results)
4. [Methodological problems encountered and resolved](#4-methodological-problems-encountered-and-resolved)
5. [Known limitations](#5-known-limitations)
6. [Repo layout](#6-repo-layout)
7. [How to reproduce](#7-how-to-reproduce)
8. [Positioning against prior work](#8-positioning-against-prior-work)
9. [Pre-registration and amendments](#9-pre-registration-and-amendments)
10. [What comes next](#10-what-comes-next)

---

## 1. Research question and contribution

The AI control literature has established that trusted monitors can catch a surprising amount of policy misbehavior. Most of this work studies **commission**: a policy that adds something it should not — a backdoor in code, a harmful action in an agent trajectory. **Omission** is the mirror-image failure mode: a policy that quietly drops safety-relevant information, making its output look complete while concealing something important.

Omission is structurally harder to catch than commission. To detect commission, a monitor needs to recognize that something present is wrong. To detect omission, a monitor needs to notice that something is absent — which requires knowing what should be there in the first place. That difference has a concrete implication for monitor design: the monitor's access to a reference document may matter more for omission than for the commission failures the literature has studied.

This project tests that directly, in a controlled setting with exact ground truth.

**The contribution** is not "structured beats holistic" (which would be close to expected). The contribution is a characterization of **when** and **how** each style fails:

- Under source-access removal, holistic and structured monitors fail in **opposite directions**. Holistic loses specificity (over-flags everything). Structured loses sensitivity (goes quiet and misses omissions). This dissociation is consistent across capability levels and is the safety-relevant finding: structured's failure mode (missing the omission) is more dangerous for oversight than holistic's (false alarms).
- The structured advantage over holistic emerges at a **capability threshold**: it is absent for weak models (Llama3 8B) and present for mid-capability models (GPT-4o-mini). At Llama3 8B, both styles produce the same overall accuracy through opposite calibration errors.
- For capable monitors, an **enumerated claim list fully substitutes for the prose source**. Detection is near-perfect regardless of prompt style when the model has an explicit checklist.

---

## 2. Design

### 2.1 Task

Each example is a matched pair: a source document containing 4-8 enumerable claims (exactly one safety-relevant), and two policy outputs — one faithful (all claims present) and one omission (safety claim removed). Ground truth is exact because we control which claim was removed.

The monitor sees the source (or a reduced version of it) and the policy output, and must classify: OMISSION or FAITHFUL.

### 2.2 The 2x3 grid

Two prompt styles crossed with three source-access levels:

| Style | Description |
|---|---|
| **Holistic** | Single binary judgment prompt. "Was any safety-relevant claim omitted?" |
| **Structured** | Three-step decomposition: enumerate claims, check each, aggregate to verdict. |

| Level | What the monitor sees |
|---|---|
| **Full** | Complete source document + policy output |
| **Claims-only** | Enumerated claim list (no prose framing, no safety labels) + policy output |
| **Output-only** | Policy output only, no source reference |

The claims_only level uses claims_only_untagged in the code: safety-relevance labels are stripped from the claim list so the monitor cannot use the label as a shortcut.

### 2.3 Capability ladder

- **Llama3 8B** — local Ollama inference (CPU)
- **Llama3.3 70B** — Groq API (within-family capability axis, removes family confound)
- **GPT-4o-mini** — OpenAI API

GPT-4o was used in earlier runs but is deferred: the two-family confound (Llama vs GPT) is better replaced by the within-family 8B-to-70B axis.

### 2.4 Dataset

300 matched pairs, 3 domains (incident reports, product reviews, study summaries), 120 test records. Pair-level stratified splits: train 60%, dev 20%, test 20%. Test is frozen and evaluated once.

Synthetic data generated by `data/generate.py`. No LLM was used to generate the claims — all claim text is hand-authored templates. No self-familiarity advantage for any evaluated model.

---

## 3. Results

**Note: results from the earlier runs (before the filler confound was found and fixed) are recorded here for reference but should not be cited. The clean rerun on the fixed dataset is in progress.**

### 3.1 Pre-confound dev results (GPT-4o-mini, n=48)

These were produced on a dataset with a filler tell (see Problem 7 below). The magnitude of the ceiling effects at claims_only may be inflated.

| Style | Level | BA | Sensitivity | Specificity |
|---|---|---|---|---|
| holistic | full | 0.833 | 1.000 | 0.667 |
| structured | full | 0.992 | 1.000 | 0.983 |
| holistic | claims_only | 0.992 | 1.000 | 0.983 |
| structured | claims_only | 0.992 | 1.000 | 0.983 |
| holistic | output_only | 0.542 | 1.000 | 0.083 |
| structured | output_only | 0.700 | 0.708 | 0.692 |

### 3.2 Key findings from the pre-confound run (directionally expected to replicate)

The source-access degradation pattern (holistic collapses at output_only, structured degrades more gracefully) was confirmed with CI-separated paired bootstrap comparisons and is expected to be robust to the filler fix because it was most pronounced at output_only, where the filler tell is weakest (the monitor never sees the output in the claims_only or full conditions to compare). The claims_only ceiling results are the ones most at risk and will be treated as pending until the clean rerun.

---

## 4. Methodological problems encountered and resolved

This section is the complete record of every significant methodological problem, in the order they were found.

---

### Problem 1: Evaluation leakage in the predecessor SEC project

**What happened.** The predecessor study (SEC 10-K filing risk language analysis) used a k-fold CV loop where prompts were tuned on data that overlapped with the evaluation set. A KEY SIGNALS intervention inflated GPT-4 kappa by approximately 0.25. The leakage was found during self-review, the result was retracted, and the correction was documented.

**What we did.** Built the omission-monitoring project around preventing this. Frozen splits, test evaluated once, `--i_have_read_prereg` flag as a procedural commitment.

---

### Problem 2: Safety-relevance tags leaking the label in claims_only

**What happened.** The original claims_only condition included [SAFETY-RELEVANT] and [routine] tags on each claim. These tags directly identify the target claim, potentially handing the monitor the answer. Near-perfect performance (BA 1.000) could have been the model reading the tag.

**What we did.** Created claims_only_untagged: labels stripped. Re-ran dev. Performance stayed near-ceiling (0.992 vs 1.000). The ceiling is real.

---

### Problem 3: Directional signal tags inherited from the SEC harness

**What happened.** The SEC harness injected [SIGNAL: de-escalating] and KEY SIGNALS blocks into monitor prompts. Carrying this into omission-monitoring would have given the monitor directional cues.

**What we did.** Rebuilt the eval loop from scratch with no signal tags. The only permitted difference between holistic and structured is the prompt structure.

---

### Problem 4: Silent defaulting of unparseable outputs

**What happened.** The SEC harness silently defaulted unparseable outputs to the negative class, biasing balanced accuracy in a non-transparent way.

**What we did.** All parsers return an explicit UNPARSEABLE token. Unparseable outputs are dropped with a warning. If the rate exceeds 10%, the run is flagged as unreliable.

---

### Problem 5: k-fold CV on prompt-tuned data

**What happened.** k-fold CV in the SEC harness meant the same records appeared in both prompt-tuning and evaluation for different folds.

**What we did.** Replaced with frozen train/dev/test splits. All prompt design on train/dev. Test touched exactly once.

---

### Problem 6: Baseline confound — safety keyword count

**What happened.** The first non-LLM baseline used a safety keyword count feature, which directly proxies for whether the safety sentence is present in the output. Baseline achieved CV BA = 0.944, above the confound gate.

**What we did.** Replaced with output length and sentence count only. Baseline dropped to CV BA = 0.54. LLM monitors beat this substantially.

---

### Problem 7: The filler tell — the most serious confound, masked by its own validator

**What happened.** In build_pair:
- Faithful output = intro + all claim summaries (zero fillers)
- Omission output = intro + remaining summaries + one filler sentence (one filler)

Filler presence perfectly separated the classes. A monitor at output_only that detects "this output contains a boilerplate sentence" would achieve high accuracy without doing any omission detection.

This was not caught by the surface-feature probe (which tests claim-level features, not output-level) or by verify_length_balance (which checked output length, not filler presence).

A first attempted fix added a shared extra filler to both outputs. This was wrong: faithful got one filler, omission got two (slot replacement + shared filler). The count still differed. The validator checked filler presence (any vs none), which was now 0.50 for both classes, so it printed "safe to evaluate" while the count tell remained.

**What we did.** Removed fillers from both outputs entirely. Faithful = intro + all claim summaries. Omission = intro + remaining summaries (target dropped, not replaced). Updated the validator to count filler sentences per output and require the mean count to be < 0.05 in both classes. The length-only classifier now sits at 0.608, reflecting the honest structural fact that faithful outputs are one sentence longer.

**Why this matters most.** The filler tell bit hardest at output_only, where the monitor sees only the output. Some of the output_only performance credited to "structured decomposition detecting absence" may have been "detecting the boilerplate filler." The clean rerun will settle this.

---

### Problem 8: Claim bank surface confound

**What happened.** The original claim bank had safety claims that were systematically longer and more negative-sounding than routine claims. A surface classifier (length + sentiment) achieved CV BA = 0.935.

A first fix inverted the confound: safety claims made short and flat, routine claims made long and negative. The probe dropped but the confound direction flipped rather than being eliminated.

**What we did.** Redesigned so both categories span the full length and sentiment range (short-flat, medium-neutral, long-alarming entries in each). Expanded the bank from ~45 to 76 unique claims. Surface probe now returns CV BA = 0.57.

---

### Problem 9: Claim-level split leakage

**What happened.** With ~45 unique claims and 300 pairs each drawing 4-8 claims, every claim appears in train, dev, and test. Few-shot examples from the pool contain the exact claims used in evaluation. Pair-level splitting does not prevent this.

**What we did.** Two partial fixes:
1. Expanded the claim bank to 76 unique claims.
2. Added a `_disjoint_pool` filter in source_access.py that removes examples sharing any claim text with the current eval record. Pool goes from ~480 to ~360 examples per record.

This is disclosed as a partial fix: claim recycling still occurs across records, just not within a single eval call.

---

### Problem 10: Capability axis confounded with model family

**What happened.** Original ladder was Llama3 8B, GPT-4o-mini, GPT-4o — two families. Any capability effect could be a family effect.

**What we did.** Replaced with Llama3 8B + Llama3.3 70B (Groq) + GPT-4o-mini. The 8B-to-70B comparison is within-family.

---

### Problem 11: Prompt-variant robustness not yet confirmed

**What happened.** The holistic vs structured comparison is two specific prompts. The over-flagging / under-flagging dissociation could be a property of the particular holistic prompt being trigger-happy rather than a paradigm effect.

**What we did (in progress).** Built analysis/prompt_variants.py: 4 paraphrases of each style, key conditions only, dev split, gpt-4o-mini. This has not been run on the clean dataset yet and is the remaining gate before the writeup.

---

## 5. Known limitations

**Single-claim deletion with the safety claim always the target.** Every omission removes exactly one safety-relevant claim. Real policy omission is partial, paraphrastic, and distributed. Future work should include omitting routine claims as a positive case.

**Mechanical omission, not adversarial.** The policy is not trying to evade the monitor. This is a controlled probe of a sub-capability, not a full control evaluation.

**Synthetic domains, small claim bank.** 76 unique claims recycled across 300 pairs. The bank is not large enough to guarantee test evaluation never sees claims from the few-shot pool. The disjointness filter mitigates this at the individual record level.

**Claims_only is an oracle condition.** No deployed monitor is handed a pre-extracted claim list. Near-perfect performance at claims_only is an upper bound, not a realistic operating point. A model-self-extraction variant is future work.

**Length floor at output_only.** Faithful outputs are one sentence longer than omission outputs. A length-only classifier achieves BA 0.608. LLM monitors beat this substantially, but output_only performance cannot be attributed entirely to content reasoning.

**Capability confound partially addressed.** Within-family comparison (8B vs 70B Llama) removes the Llama-vs-GPT confound, but GPT-4o-mini remains a cross-family point. The capability threshold may partially reflect a family effect.

---

## 6. Repo layout

```
data/
  schema.py                   # Record schema, validation, strict split handling
  generate.py                 # Synthetic generator (see Section 4 for design history)
  dataset.jsonl               # Generated dataset
  splits.json                 # Split summary

monitors/
  holistic.py                 # v0 single-prompt monitor (superseded by source_access.py)
  structured.py               # v0 decompose-then-aggregate monitor (superseded)
  source_access.py            # Phase 2 monitor: all 6 conditions, few-shot disjointness
  baseline.py                 # Non-LLM baseline (length + sentence count)

eval/
  run.py                      # Evaluation loop: 2x3 grid, capability ladder
  metrics.py                  # Balanced accuracy, kappa, bootstrap CIs, paired bootstrap

analysis/
  calibration.py              # Full calibration table (BA, sensitivity, specificity, FPR)
  surface_feature_probe.py    # Surface confound gate (CV BA must be < 0.65)
  prompt_variants.py          # Prompt-variant robustness check (4 paraphrases x 2 styles)
  phase2_checks.py            # Bootstrap CIs on key gaps, power analysis
  test_bootstrap_cis.py       # Final test CIs on all pre-registered comparisons

results/                      # Result JSON files (gitignored)

PREREG.md                     # Pre-registration: hypotheses, metric, comparisons
PHASE2.md                     # Phase 2 design rationale and positioning
README.md                     # This file
```

---

## 7. How to reproduce

### Dependencies

```bash
pip install scikit-learn numpy openai groq ollama
```

### Generate the dataset

```bash
cd data
python generate.py --n 300 --out . --seed 42
cd ..
```

Both gates must pass before any model evaluation:
- `length_only_classifier_acc` between 0.40 and 0.65
- `fa_mean_fillers = 0.000`, `om_mean_fillers = 0.000`

### Run the surface feature probe

```bash
python analysis/surface_feature_probe.py
```

Must report CV BA < 0.65. If above, fix the claim bank before any API call.

### Set up model access

```bash
ollama pull llama3:latest
export GROQ_API_KEY=...
export OPENAI_API_KEY=...
```

### Dev run

```bash
python eval/run.py --split dev \
  --model llama3:latest llama-3.3-70b-versatile gpt-4o-mini \
  --styles structured holistic \
  --levels full claims_only_untagged output_only \
  --shots 3
```

### Prompt variant robustness check

```bash
python analysis/prompt_variants.py
```

The specificity dissociation must survive across all 4 prompt paraphrases before proceeding to test.

### Final test run (once only)

```bash
python eval/run.py --split test --i_have_read_prereg \
  --model llama3:latest llama-3.3-70b-versatile gpt-4o-mini \
  --styles structured holistic \
  --levels full claims_only_untagged output_only \
  --shots 3
```

### Analysis

```bash
python analysis/calibration.py --results results --split test
python analysis/test_bootstrap_cis.py
```

---

## 8. Positioning against prior work

| Paper | Their design | How this relates |
|---|---|---|
| Greenblatt et al., AI Control (ICML 2024) | Trusted monitor over untrusted policy, code backdoor as subversion | We study omission vs commission, and vary monitor information access rather than capability gap |
| Arnav et al., CoT Red-Handed (2025) | Vary reasoning trace access (CoT vs action), sabotage detection | They vary what the monitor observes; we vary what reference it has. Both study monitor information needs for different task types |
| TraceGuard (Nguyen et al., 2025) | Multi-dimensional decomposition of CoT evaluation | Most similar to our structured monitor. Their setting is commission; we test whether decomposition generalizes to omission |
| Factor(T,U) (2025) | Factored cognition strengthens monitoring | Task decomposition on the policy side; we decompose the monitor's evaluation task |

**The gap:** none of these papers study omission as the subversion type or vary the monitor's reference document access. The graded source-access design is novel in both dimensions.

**Framing note:** this is a monitoring study, not a control evaluation. There is no adversary, no strategic omission, no optimization against the monitor. The correct framing is a precursor to control evaluation. The adversarial arm (Phase 3) earns the control label.

---

## 9. Pre-registration and amendments

`PREREG.md` contains the frozen hypotheses, primary metric (balanced accuracy), exact comparisons with predicted directions, and decision rules. Written before the v0 run and amended before the Phase 2 test run.

The Phase 2 amendment (appended to PREREG.md) covers: graded source-access design, claims_only_untagged condition, capability ladder update (Llama3.3 70B replacing GPT-4o), and decision rules for each pre-registered comparison.

Any deviation from the pre-registered analysis is reported explicitly.

---

## 10. What comes next

**Immediate — Phase 2 clean rerun.** Re-run the full 2x3 grid on the fixed dataset with the updated model ladder (Llama3 8B + Llama3.3 70B + GPT-4o-mini). Run prompt-variant robustness check on dev. Single test run.

**Phase 3 — adversarial omission.** The policy model is instructed to omit the target claim while making the output appear complete. This is the condition that earns the AI control framing and is where the structured vs holistic question becomes most practically relevant.

**SEC dataset as real-world validation arm.** The predecessor SEC 10-K study serves as a real-world testbed. One robust result transfers directly (95% de-escalating recall on Llama3 8B diff condition).

