# Day 1 novelty memo — five flagged papers + fresh-search check

One page per paper: what it contributes, and what it leaves open relative to
our framing (per-query, 5-action {STOP, SAMPLE, SELECT, SEARCH, ABSTAIN}
prediction from a cheap k=4 probe, at *matched token* cost including
discarded search branches, for a small non-thinking policy, validated
against a pre-hoc query-text-only comparator and an oracle).

---

## 1. [2604.17433 — Self-Consistency from Only Two Samples: CoT–PoT Ensembling for Efficient LLM Reasoning](https://arxiv.org/html/2604.17433)

CMU Qatar / QCRI, June 2026. Ensembles Chain-of-Thought and Program-of-Thought
outputs from as few as two samples to approximate full self-consistency
cheaply, exploiting the fact that the two reasoning modes fail on different
problems.

**What remains open:** This is a *fixed* ensembling recipe (always combine
one CoT + one PoT sample) — it is not a per-query decision among several
qualitatively different actions, has no notion of spending a *variable*
downstream budget, and has no abstention. It targets "cheap self-consistency"
specifically, not "which purchase pays." No overlap with the action-value
prediction question.

## 2. [2606.09078 — The Hidden Bias of Process Reward Models: PRISM for Rewarding the Right Reasoning](https://arxiv.org/abs/2606.09078)

June 2026. Identifies that PRM training data is step-imbalanced, which makes
standard cross-entropy training overcredit plausible-but-wrong steps
(elevated false-positive rate). Proposes PRISM, a contrastive step-level
training scheme with hard negatives from temporal lookahead — no new human
labels required.

**What remains open:** This is squarely about *training a better PRM*, which
is out of scope for us by invariant #9 (no PRM training/fine-tuning). It
doesn't touch per-query action prediction at all. It's directly relevant as
background for our V4 (PRM length/step bias) supporting analysis and as a
citation for *why* PRM-argmax might behave oddly — but it competes with
nothing in H1–H4.

## 3. [2606.08098 — When Does Delegation Beat Majority? A Delegation-Based Aggregator for Multi-Sample LLM Inference](https://arxiv.org/html/2606.08098)

Sakai, Song, Larson; updated July 2026. Proposes Propagational Proxy Voting
(PPV): each sample-cluster keeps weight on its own answer by entropy-based
confidence and routes remaining weight to peers by reasoning-embedding
similarity — a training-free, label-free alternative to majority voting.
+1.5pp over majority on MMLU-Pro at N=128 (+2.24pp on non-trivial questions).

**What remains open:** This is a new *selector* — machinery inside what we'd
call the SELECT action, evaluated only in aggregate over a fixed N. It is not
a per-query predictor of *which* action to take, carries no budget/token
accounting, and has no abstention. At most it's a candidate SHOULD-tier
selector variant (verifier-free selector family) for us — it doesn't compete
with the action-prediction claim.

## 4. [2607.08065 — When LLMs Agree, Are They Right? Auditing Self-Consistency and Cross-Model Agreement as Confidence Signals](https://arxiv.org/abs/2607.08065)

Ding (UPenn), July 2026. Large audit (53 runners × 50 samples, 265k samples
total, GPQA Diamond + AIME) showing agreement ≠ correctness: models can agree
with themselves or each other from shared bias/memorized heuristics/
position priors rather than truth.

**What remains open:** This is a negative/auditing result about agreement as
a *confidence signal*, not a proposal of any predictor or controller. It's
directly relevant motivation for us — it's a caution that our "agreement"
feature group (top-1 vote fraction, entropy, margin) may be a noisier signal
than assumed, which is exactly why the feature ablation (MUST) and the
pre-hoc comparator matter. It doesn't propose or test action allocation, PRM
selection, or search, so no novelty overlap — but it strengthens the case for
why H2 needs the confidence-group ablation and the V5 reimplementation.

## 5. [2506.12721 — Strategic Scaling of Test-Time Compute: A Bandit Learning Approach](https://arxiv.org/pdf/2506.12721)

Zuo & Zhu, UC Riverside; to appear ICLR 2026. Already in brief.md's prior-work
table. Formulates test-time compute allocation as a bandit-learning problem;
estimates query difficulty online and allocates more compute to
hard-but-solvable queries, provably beating uniform allocation, validated on
math and code benchmarks.

**What remains open (confirmed, matches brief.md §7):** Predicts *difficulty*,
not *which action* to take — no PRM-based SELECT vs. bounded SEARCH
distinction, no abstention as a first-class action, and allocation is in
generic "compute units" rather than exact matched tokens (no charge for
discarded search branches). Our gap — per-query action-type prediction, not
just amount — survives untouched.

---

## Fresh-search check for duplicates of the per-query-action-prediction framing

Beyond the five flagged papers, I ran an explicit search for anything that
predicts *which* test-time-compute action to take (not just how much) from
cheap per-query evidence. One close candidate surfaced:

### [2604.14853 — Adaptive Test-Time Compute Allocation for Reasoning LLMs via Constrained Policy Optimization](https://arxiv.org/abs/2604.14853)

Formalizes compute allocation as a constrained optimization (maximize
expected accuracy subject to an average compute budget), solved via a
two-stage "Solve-then-Learn" pipeline: Lagrangian relaxation finds oracle
actions pricing accuracy against cost, then a lightweight classifier learns
to predict those oracle actions from cheap input features, amortizing the
allocation rule for deployment. Up to 12.8% relative accuracy gain on MATH
under matched budget, tested on DeepSeek-V3, GPT-4o-mini, and Qwen2.5-7B on
MATH/GSM8K.

**This is the closest paper found and is worth tracking**, but it is
distinguishable on every axis that matters to our framing:
- Its "actions" are generic compute-scaling knobs (more sampling / search /
  extended reasoning) priced by a single Lagrangian multiplier, not our
  discrete typed action set with a PRM-defined SELECT distinct from a
  bounded-search SEARCH.
- No abstention action.
- No pre-hoc query-text-only control baseline (the paper's baselines are
  "uniform and heuristic allocation" plus the Lagrangian oracle) — so it
  can't falsify an H3-style claim.
- No stated exact token accounting for discarded search branches.
- Model mix is frontier + mid-size (DeepSeek-V3, GPT-4o-mini, Qwen2.5-7B),
  not a small non-thinking policy under matched-token discipline.

**Verdict: framing survives.** No paper found (flagged or fresh-search)
predicts *which* of {STOP, SAMPLE, SELECT, SEARCH, ABSTAIN} to take per
query from a cheap probe, at matched token cost with discarded-branch
accounting, for a small model, against both a pre-hoc and an oracle
baseline. 2604.14853 should be cited in `report/01-literature.md` as the
closest related work.

**Caveat on source quality:** paper summaries above come from arXiv listing
pages, HTML renders, and one abstract fetch — not a full close read of PDFs
(Day 1's 7h budget doesn't cover that). Day 2's literature review
(`report/01-literature.md`) should verify claims against the full text
before they're relied on for the report's prior-work table.
