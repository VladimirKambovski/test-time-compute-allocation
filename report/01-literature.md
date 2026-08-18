# Literature Review, Motivation, Research Question, Prior Work

_Drafted Day 2 per docs/roadmap.md. Content synthesized from docs/brief.md
§1, §5, §7, and `notes/2026-08-18-novelty-memo.md`; will be extended as
DIAGNOSE/PREDICT/ALLOCATE results land in Weeks 2-3._

## Research question

Can a small reasoning model use cheap early evidence — a four-sample probe
drawn before any expensive test-time compute is spent — to predict how its
*remaining* budget should be allocated, and thereby reach a better
quality-compute tradeoff than any fixed allocation policy?

A system serving a small reasoning model under a fixed token budget must
decide, per request, what to buy: more independent samples (SAMPLE), process-
reward-model scoring of fewer samples (SELECT), bounded verifier-guided
search (SEARCH), or nothing at all — either because the probe already
answers the question (STOP) or because no affordable spending is likely to
help (ABSTAIN). This project asks whether the evidence already sitting in a
cheap four-sample probe is enough to tell which of these five actions pays,
per query.

## Motivation

The default recipe for improving a small model's accuracy at test time —
sample N times and vote — spends an identical budget on every request,
regardless of whether spending helps. Two distinct failure modes hide behind
the single word "hard":

1. The correct answer is already present among the samples, but plain
   majority voting picks the wrong one — here, *spending on selection*
   would help, not spending on more samples.
2. The correct answer never appears at any budget the system can afford —
   here, *no spending helps*, and the honest behaviour is to decline and
   route elsewhere.

Existing adaptive-compute work largely predicts *difficulty*, which
conflates these two cases: a "hard" query might be hard-but-solvable (case
1) or hard-and-unsolvable (case 2), and a difficulty score alone doesn't
distinguish them. Coverage-scaling work (e.g. Large Language Monkeys)
measures the sampling-vs-selection gap in aggregate, across a whole
benchmark, without predicting it per request.

For a team running a small or local model under cost, latency, or privacy
constraints, the missing capability is a principled, per-query answer to
*when to stop paying, and for what.* An abstention that says "not
recoverable locally, route upstream" converts a silent quality failure into
a deliberate, machine-readable routing event — this is the Compute-Aware
Reasoning Gateway framing this project delivers as its applied artifact.

## Prior work and the remaining gap

| Work | Contribution | Leaves open |
|---|---|---|
| Large Language Monkeys (2407.21787) | Coverage (pass@k) scales log-linearly with sample count while selection (maj@k) plateaus much earlier | Aggregate, large-model result; no per-query prediction of which regime a given request is in |
| Compute-optimal TTS (2502.06703) | The optimal test-time strategy depends jointly on policy strength, PRM choice, and problem difficulty | Matches *sample count*, not tokens, across strategies; no per-query predictor — a difficulty-conditioned policy is proposed, not learned from cheap evidence |
| Adaptive-Consistency (2305.11860); Dynasor; DEER; SEER | Cheap agreement/consistency signals let you stop sampling early at ~zero accuracy cost | All predict *when to stop*, never *which action to take instead*, and none includes abstention as a first-class outcome |
| Zuo & Zhu, Strategic Scaling of TTC (2506.12721) | Formulates compute allocation as bandit learning; estimates difficulty online, allocates more to hard-but-solvable queries, provably beats uniform allocation | Predicts difficulty/amount, not action *type*; no PRM-based SELECT vs. bounded SEARCH distinction; no abstention; compute measured in generic units, not exact matched tokens |
| Best-of-Majority (2510.03199); PRISM (2606.09078) | Best-of-N regret theory; diagnoses PRM length/step-count bias from training-data imbalance | Neither offers an empirical per-query action-value map for small policies; PRISM is about *training* a better PRM, which is out of this project's scope |
| Adaptive TTC Allocation via Constrained Policy Optimization (2604.14853) — found via Day-1 fresh-search novelty check, not in the original five flagged papers | Learns a classifier to predict Lagrangian-priced oracle actions (generic compute-scaling knobs: more sampling / search / extended reasoning) from cheap features, on DeepSeek-V3, GPT-4o-mini, Qwen2.5-7B | Actions are a single priced knob, not a typed action set with PRM-defined SELECT distinct from bounded SEARCH; no abstention; no pre-hoc query-text control (so it can't falsify an H3-style claim); no exact token accounting for discarded search branches; not focused on a small non-thinking policy |

**The gap this project targets:** per-query prediction of *which* compute
action pays — not just how much, and not just when to stop — for a small
open-weight model, at matched token cost (charging discarded search
branches, not just emitted tokens), evaluated against both a pre-hoc
query-text baseline and an oracle, with abstention as a first-class,
equally weighted action alongside spending more.

See `notes/2026-08-18-novelty-memo.md` for the full one-page-per-paper
novelty check (five flagged papers plus the fresh-search duplicate check)
that established this gap survives current literature as of 2026-08-18.

## Research flow

| Stage | Question | Produces | Hypothesis |
|---|---|---|---|
| DIAGNOSE | Does the allocation choice matter, and is it heterogeneous across queries? | The action-value landscape | H1 |
| PREDICT | Can cheap early evidence identify the winning action? | Predictor + comparators | H2, H3 |
| ALLOCATE | Does acting on the prediction improve the quality-compute frontier? | Pareto frontier vs. fixed policies | H4 |

Full hypothesis statements and accept conditions: docs/brief.md §10.

## Contribution, stated conservatively

No new algorithm. The contribution is a per-query action-value landscape for
a small open-weight model at matched token cost; a reformulation of that
landscape as a prediction problem from cheap probe evidence, evaluated
against a pre-hoc baseline and an oracle; and a reusable research system
where one controller drives both offline replay and a live gateway,
verified by a parity test. Workshop-paper scale, as stated in docs/brief.md
§6.
