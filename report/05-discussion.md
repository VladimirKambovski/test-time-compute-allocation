# Discussion (Verified vs. Speculative)

## What's verified — solid, multiply-confirmed

**There is no meaningful test-time-compute allocation headroom for this
policy, at this problem difficulty range, under the frozen decode config.**
This is the project's central negative finding, and it is unusually
well-supported for a project of this scale:

- Confirmed on both dev benchmarks independently (MATH-500 and
  OlympiadBench-A).
- Confirmed at full P1 scale with a 95% CI that excludes zero — the gap is
  real, just tiny (~1pp against an 8pp accept bar).
- **Confirmed a second, independent time on held-out data never touched
  during development** — a stronger validity check than most negative
  results in this space get. The held-out gap is 0.00pp, not just "small."
- Confirmed to be robust to the token budget specifically — the mentor's
  own suggested check found the gap is *identical* (1.89pp) at 4x the
  frozen budget, with a paired-difference CI that clearly includes zero.
  This rules out the most obvious alternative explanation (an artificially
  short decode budget suppressing real headroom).

**SELECT (PRM-weighted answer selection) provides essentially no value on
this policy/data combination**, confirmed 5 independent ways spanning
different mechanisms (oracle ceiling, weaker-policy diagnostic, real PRM
reproduction, temperature ablation targeted at known SELECT cases,
max_tokens ablation on the same cases) plus a 6th line of evidence (the
simplest possible PRM-argmax selector is reliably *worse* than plain
majority, not just no-better, and the deficit grows with more samples).

**The predictor (Detective) has genuine discriminative signal** — macro-
AUROC 0.87, real held-out generalization (the frozen model was evaluated
once on data it never saw during fitting), and a real, substantial margin
over the pre-hoc control (H3, 4x the required threshold). This is not an
artifact of class imbalance: a trivial majority-class baseline is
mechanically fixed at AUROC 0.5, and Detective clears that by a wide
margin on every class except SELECT (which has too few examples, 6 total,
for any method to learn from reliably).

## What's verified but more consequential than initially framed

**On held-out data, deploying the learned controller would cost accuracy
relative to doing nothing clever at all** (-9.3pp vs. always sampling on
P4). This is a stronger, more specific claim than "the controller doesn't
help" (the dev-side H4 finding), and it's the single most important result
for anyone considering deploying a system like this: a good macro-AUROC
number does not guarantee a good real-world outcome when the class
imbalance and the achievable-value ceiling are this skewed. The mechanism
is traceable (Detective over-abstains relative to the true optimal
distribution on this specific held-out set), which makes it a diagnosable,
not mysterious, failure — but it is a failure that a report focused only
on AUROC would have missed entirely.

## What's real but more limited in scope than it first appears

**The band-0 "unwinnable" / "carries the gap" characterization was
partially a token-budget artifact.** This is a genuine finding, not an
error — but it applies specifically to how the difficulty-band *story* was
told, not to the headline G1/SELECT verdicts (which are decided by the
60% of problems in the clean, unaffected bands). The corrected
characterization — "roughly 5/6 of the hardest tail recovers with more
budget, roughly 1/6 has a real ceiling" — is itself well-evidenced (n=30
systematic sample, not the original n=6 hand-picked one), but it is a
secondary, descriptive claim, and readers should not conflate it with a
challenge to the aggregate results.

**V3 (self-consistency plateau/decline on hard items) is only partially
tested.** The aggregate stabilization/plateau evidence is solid (k*≈2.17,
a monotonically increasing majority-accuracy curve with no aggregate
decline); the specific per-difficulty-band decline-after-peak question the
literature claim asks about was not run, a deliberate time-pressure cut
disclosed in `03-verification.md` and `07-next-steps.md` rather than
silently skipped.

## Speculative / not directly tested

- **The mechanism connecting V4 (PRM-argmax winners are shorter but have
  more steps) to the truncation finding** — plausible (PRM favors samples
  that actually finished over ones cut off mid-derivation, which
  correlates with being shorter overall but completing their full step
  sequence) but not independently proven. Flagged as a hypothesis, not a
  demonstrated causal claim.
- **Whether class-weighted rebalancing is the right production choice for
  Detective.** The full-rebalancing trade-off (SAMPLE recall up, ABSTAIN
  recall and overall accuracy down) is solid; which side of that trade-off
  is "better" depends on the deployment's actual cost function for a wrong
  ABSTAIN vs. a missed SAMPLE opportunity, which this project does not
  have real data on. **Update, tested rather than left speculative**: a
  more targeted, partial version of this fix (not the full extreme) was
  tried against held-out — three independently-selected variants (partial
  reweighting, a confidence-threshold rule, and a combined version) all
  converge on the same modest recovery, -9.30pp → -8.14pp vs. best fixed.
  Real, but partial — see `04-results.md` and `notes/2026-09-01.md`.
- **Generalization to other policy sizes or genuinely official model
  weights.** All results come from one 4B model served through a
  third-party GGUF conversion (see below) and one weaker-policy (2B)
  diagnostic used only to corroborate the SELECT-rarity finding, not
  independently validated at full scale.

## A methodology caveat worth stating plainly, not deferred

**Every real number in this project comes from a third-party `unsloth`
GGUF conversion of the primary policy model, served via `llama.cpp` — not
the official Hugging Face safetensors weights via vLLM.** This was
identified and documented internally from Day 3 (`configs/policies/qwen3.5-4b.yaml`
carries the caveat directly) but was not raised with the mentor in plain
language until late in the project. It does not appear to have introduced
an obvious artifact — the golden-200 hand-check, the k*≈2.17 stabilization
finding, and the held-out replication all behave as expected of a real,
functioning instruction-tuned model — but it is a real, disclosed
provenance gap between what was tested and the officially-published
checkpoint, and any claim of exact reproduction against papers that used
the official weights should be read with that caveat attached.

## A real methodology bug, found and fixed, disclosed here for completeness

A duplicate-pool-directory issue (49 problems had more than one valid
on-disk pool, from what appear to be historical retry artifacts) caused
several aggregate analysis scripts to double- or triple-count some
problems. Caught mid-session by noticing an unexplained discrepancy in a
problem count, traced, and fixed by switching every aggregate script to
canonical, content-addressed pool enumeration rather than a directory
glob. The corrected numbers (reported throughout this report) are what
actually shipped; the pre-fix numbers were briefly reported internally and
are documented as superseded in `notes/2026-08-26.md`, not silently
replaced. Both H2 and H3's verdicts were unaffected by the fix (H3's
margin actually increased after correction), which is itself evidence the
underlying finding is robust rather than an artifact of the bug.
