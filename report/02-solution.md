# Solution and Methodology

## System overview

The system is a **Compute-Aware Reasoning Gateway**: given a query, it draws
a cheap 4-sample probe from the policy model, extracts a small set of
backend-independent features from that probe, and uses a learned or fixed
controller to choose one of four actions — **STOP** (answer with the probe's
own majority, spend nothing more), **SAMPLE** (spend the remaining budget on
more independent samples, plain majority), **SELECT** (spend the same budget
split between more samples and PRM scoring, PRM-weighted majority), or
**ABSTAIN** (decline, spend nothing, report a machine-readable reason). A
fifth action, **SEARCH** (bounded PRM-guided beam search), was in the
original design and frozen action space but was dropped mid-project after a
real, unfixed continuation-strategy bug (see `06-retrospective.md`).

One `Controller.decide()` implementation drives both the offline replay
engine and the live `/solve` gateway — enforced by
`tests/test_controller_parity.py`, which asserts byte-identical decisions
through both entry points. This is the one architectural invariant the
whole demo depends on; the file that would break it (`gateway/app.py`
containing its own allocation logic instead of delegating) is exactly what
that test is designed to catch.

## Models

| Role | Model | Note |
|---|---|---|
| Primary policy | `Qwen/Qwen3.5-4B`, non-thinking mode | Served via a third-party `unsloth` GGUF conversion through `llama.cpp`, **not** the official HF safetensors via vLLM — flagged internally from Day 3, disclosed to the mentor. Every real number in this project comes from that hosted conversion. |
| Primary PRM | `Qwen/Qwen2.5-Math-PRM-7B`, scalar step-level scorer | Real hosted endpoint, zero failures across thousands of calls project-wide. |

Both revisions are pinned (see `configs/policies/qwen3.5-4b.yaml`,
`configs/prms/qwen-math-prm-7b.yaml`); any change to either — model
revision, quantization, temperature, top_p, top_k, max_tokens, prompt
template, stop sequences, or backend/provider — creates a new pool, never a
silent continuation of an existing one (`pools/store.py::compute_pool_id`,
a content-address over exactly those fields).

## Data

| Pool | Role | Size | Status |
|---|---|---|---|
| P1 | Dev, primary | MATH-500, 500 problems, N=32 | fully generated + PRM-scored |
| P2 | Dev, secondary | OlympiadBench slice A, 300 problems, N=32 | fully generated + PRM-scored |
| P4 | Held-out, in-distribution | OlympiadBench slice B, 100 problems, N=32 | generated once, scored, evaluated |
| P5 | Held-out, out-of-distribution | AIME25, 30 problems, N=32 | generated once, scored, evaluated |

N=32 is the MUST floor (changed from an original N=64/N=128 design,
2026-08-20, by explicit instruction — see `notes/2026-08-20.md`). Every
pool is a **nested-prefix store**: all N samples are generated once and
stored; any analysis at a smaller budget level reads a prefix of the same
pool rather than regenerating, so every budget-level comparison is
perfectly paired. Held-out sets were frozen on Day 2 (SHA-256 hash of the
sorted problem-id list, committed to git) and never touched before the
frozen controller's single evaluation pass on Day 18 — re-verified
byte-identical to the frozen hash immediately before that pass.

## Features (the probe)

Computed from the first 4 samples only — `decide()` never conditions on
more than the free probe, since anything else would leak the budget's own
future allocation into the decision that chooses it:

- **Agreement** (backend-independent): top-1 vote fraction, top-2 margin,
  normalized entropy, distinct-answer count.
- **Shape** (backend-independent): mean/variance of output length, mean
  step count.
- **Hygiene** (backend-independent): extraction-failure fraction,
  truncation fraction.
- **Confidence** (requires per-token logprobs, backend-dependent): mean/min
  logprob, mean self-certainty, cumulative-logprob spread. `NaN`, never a
  silently-wrong 0, when unavailable (P1 has none — generated before a
  Day-10 backend fix that started requesting them; P2 and everything after
  has real per-token logprobs).

## The oracle label and the controller

The oracle action label `a*(q,B)` is computed per problem from the full
32-sample pool and the gold answer: STOP if the 4-sample probe's own
majority is already correct, else SAMPLE if the full-pool majority is
correct, else SELECT if a correct answer exists anywhere in the pool even
though the majority missed it, else ABSTAIN. This is a true 4-class label
(STOP/SAMPLE/SELECT/ABSTAIN), matching the brief's literal spec — an
earlier draft that hard-excluded SELECT from the label space was reverted
in favor of letting SELECT's near-zero rate show up as a measured result
of the data, not an a priori exclusion (see `notes/2026-08-23.md`).

**Detective** (the real, learned predictor) is a multinomial logistic
regression on the probe features, fit with grouped-by-problem,
stratified-by-benchmark 5-fold CV (`sklearn.model_selection.StratifiedKFold`,
`random_state=20260826`), then frozen for deployment as a single final fit
on the full 754-problem canonical dev set
(`results/models/detective_frozen.joblib`) — never refit on held-out data.

**Fortune Teller** (the pre-hoc control) is the same model family fit on a
`sentence-transformers` embedding of the query text alone, before any probe
sample exists — the non-negotiable comparator that makes H3
("post-hoc evidence beats a pre-hoc guess") falsifiable.

Five additional fixed comparators (Miser=always-STOP, Spendthrift=always-
SAMPLE, UniformSelect=always-SELECT, Gambler=random at a fixed rate,
Oracle=the true per-budget ceiling) and two more learned/simple comparators
(majority-class floor, a fixed agreement-threshold heuristic) complete the
E7 policy set (see `results/heldout_results.json`,
`notes/scratch/day15_e7_e8_pareto_corrected.py`).

## Statistics

Every comparison is a per-problem paired difference. 10,000-resample BCa
(bias-corrected and accelerated) bootstrap CIs for every reported gap
(`evaluation/stats.py::paired_bootstrap_bca`); McNemar for paired binary
correctness flips at fixed budget; Holm-Bonferroni within declared
comparator families; no mean/CI reported for any cell with fewer than 5
replicates (the honesty rule) — medians and ordinal statements instead.
P5 (AIME25, n=30) is reported this way throughout: raw correct/total
counts, no confidence interval dressed up with false precision.

## Budget accounting

SAMPLE and SELECT receive the **same** nominal budget B. Because PRM
forward passes are not free, SELECT buys strictly fewer raw samples than
SAMPLE at equal B — the split is computed by
`budget/accounting.py::budget_split_for_select`, and every charge (policy
tokens, PRM forwards, discarded search-branch tokens) is tracked
separately and summed into a token-equivalent for cross-action comparison.
One disclosed, unavoidable modeling assumption: no PRM-forward-cost figure
is documented anywhere in the frozen spec or PRM config, so a PRM forward
is assumed to cost the same token-equivalent as one policy sample's
generation — a conservative, easy-to-state convention, not a measured
number, flagged for the mentor rather than silently settled.

## Answer checking

`math_verify` (or an equivalent, never a hand-rolled regex) for
extraction/equivalence. Four real correctness bugs in how it was being
used were found and fixed early (Day 3, golden-200 hand-check — silent
set-construction on disagreeing boxed answers, a truncation-fallback
ordering bug, bare-LaTeX mis-parsing on both sides). A closed failure
taxonomy (`answers/taxonomy.py`) records every extraction/scoring failure
mode explicitly; an unrecognized status fails loudly rather than being
coerced into one of the known ones.
