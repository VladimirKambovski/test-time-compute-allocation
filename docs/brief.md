# Project Brief — FINAL (v2)

# Predicting Where the Next Token Should Go

### Evidence-driven test-time compute allocation for small reasoning models, delivered as a Compute-Aware Reasoning Gateway

**Status: scope pre-registered and frozen subject to predefined gates G0–G10.** Changes triggered by those gates follow the fallback paths already written into this document. Any other material change requires a mentor discussion and a dated entry in `notes/`.

**Revision note (v2).** No change to the research question, the Diagnose → Predict → Allocate structure, H1–H4, the controller concept, or the system. v2 changes only feasibility floors, execution ordering, and the compute plan: N=64 is the guaranteed floor, the search arm's committed workload is reduced, the 32B comparator moves to SHOULD, gateway integration and report writing start in Week 2, a MUST-before-SHOULD execution rule is added, program-process deliverables are made explicit, and the compute plan is rewritten to be backend-agnostic rather than assuming a dedicated GPU.

---

## 1. Primary research question

> **Can a small reasoning model use cheap early evidence to predict how its remaining test-time compute should be allocated, and thereby achieve a better quality–compute tradeoff than fixed allocation policies?**

## 2. Summary

A system serving a small reasoning model under a fixed token budget must decide, per request, what to buy: more independent samples, process-reward-model scoring of fewer samples, bounded verifier-guided search, or nothing. This project draws a cheap four-sample probe and asks whether the evidence in that probe predicts which purchase pays.

**Diagnose** establishes that the choice matters — that different actions win on different queries and that oracle per-query allocation beats the best single fixed policy. **Predict** tests whether cheap probe evidence identifies the winning action, against a pre-hoc query-text classifier, a published agreement heuristic, and an oracle. **Allocate** puts the predictor into a controller and measures the quality–compute Pareto frontier against fixed policies at matched token budgets.

All sampling is generated once into content-addressed frozen pools with nested prefixes, so every comparison is perfectly paired and all downstream analysis replays at zero inference cost. The controller is deployed as a Compute-Aware Reasoning Gateway that answers cheaply, escalates deliberately, or declines — with the same controller object driving offline evaluation and the live demo, enforced by a parity test.

## 3. The research flow

| Stage | Question | Produces | Hypothesis |
|---|---|---|---|
| **DIAGNOSE** | Does the allocation choice matter, and is it heterogeneous across queries? | The action-value landscape | H1 |
| **PREDICT** | Can cheap early evidence identify the winning action? | Predictor + comparators | H2, H3 |
| **ALLOCATE** | Does acting on the prediction improve the quality–compute frontier? | Pareto frontier vs fixed policies | H4 |

**How every retained component serves the flow** — unchanged from v1:

| Component | Stage | Role |
|---|---|---|
| Frozen nested candidate pools | all | Makes the landscape computable at every budget without regeneration; makes comparisons paired |
| pass@k vs maj@k | DIAGNOSE | Bounds the headroom sampling-plus-selection can reach |
| PRM scoring and selection | DIAGNOSE | Defines the SELECT action |
| Bounded guided search | DIAGNOSE | Tests whether SEARCH occupies a distinct region of the landscape |
| Matched-token accounting | DIAGNOSE, ALLOCATE | Actions are comparable only if charged identically |
| Pre-hoc comparator | PREDICT | The control: does sampled evidence beat query text? |
| Post-hoc predictor | PREDICT | The contribution |
| Paired evaluation | all | Statistical power at n≈800 without many seeds |
| Validity gates | all | Protection against silent correctness failures |
| Shared Controller + parity test | ALLOCATE, demo | The served policy *is* the evaluated policy |
| Reasoning Gateway + demo | ALLOCATE | The finding operationalized |

## 4. The action space

After a probe of k=4 samples, with remaining budget B tokens:

| Action | Behaviour | Marginal cost |
|---|---|---|
| **A0 STOP** | Answer with the probe majority | 0 |
| **A1 SAMPLE** | Spend all of B on more samples, then plain majority | B policy tokens |
| **A2 SELECT** | Spend part of B on more samples and part on PRM scoring, then PRM-weighted selection | B split between policy tokens and PRM forwards |
| **A3 SEARCH** | Spend B on bounded PRM-guided beam search | policy tokens incl. discarded beams + PRM forwards |
| **A4 ABSTAIN** | Spend nothing; decline and report why | 0 |

A1 and A2 receive the *same* budget. Because PRM forwards are not free, A2 buys fewer samples than A1 — so "sample more" versus "select better" is a genuine matched-token tradeoff. This is what makes exact token accounting load-bearing rather than decorative.

**Oracle action** `a*(q, B)` = the cheapest action yielding a correct answer, or A4 if none does.

**Label scope.** A3 labels exist only on the search subset. Primary predictor: **4-class** (A0, A1, A2, A4) over all ~800 dev problems. Secondary: **5-class** on the search subset, with an explicit small-*n* caveat. If DIAGNOSE finds A3 never wins, the action space is 3-way plus abstain, the controller simplifies, and that is a reportable negative result.

## 5. Motivation

The default recipe — sample N times and vote — spends identically on every request regardless of whether spending helps. Two failure modes hide under "hard": the correct answer is already among the samples but voting picks wrong (spending helps), or the correct answer never appears at any affordable N (spending is waste). Existing adaptive-compute work predicts *difficulty*, conflating them; coverage-scaling work measures the distinction in aggregate without predicting it per request.

For anyone running a small or local model under cost, latency, or privacy constraints, the missing capability is a principled answer to *when to stop paying*. An abstention that says "not recoverable locally, route upstream" converts a silent quality failure into a deliberate routing event.

## 6. Contribution, stated conservatively

No new algorithm. The contribution is:

1. A **per-query action-value landscape** for a small open-weight model, measured at matched token cost across sampling, PRM selection, and bounded search.
2. **Reformulating that landscape as a prediction problem** from a cheap probe, with an explicit label definition and evaluation against a pre-hoc query-text baseline, a published agreement heuristic, and an oracle.
3. A **reusable research system** where one controller drives both offline replay and a live gateway, verified by a parity test.

Workshop-paper scale. The report says so.

## 7. Prior work and the remaining gap

| Work | Contribution | Leaves open |
|---|---|---|
| Large Language Monkeys (2407.21787) | Coverage scales log-linearly while selection plateaus | Aggregate, large models; no per-query prediction |
| Compute-optimal TTS (2502.06703) | Optimal strategy depends on policy, PRM, difficulty | Matched *sample* count, not tokens; no per-query predictor |
| Adaptive-Consistency (2305.11860); Dynasor; DEER; SEER | Cheap-signal early stopping | Predicts *when to stop*, never *which action*, never abstention |
| Zuo et al. (2506.12721) | Budget allocation as bandit learning with difficulty estimation | Difficulty, not action choice; no abstention |
| Best-of-Majority (2510.03199); PRISM (2606.09078) | BoN regret theory; PRM length bias | No empirical action-value map for small policies |

**Gap:** per-query prediction of *which* compute action pays, for a small model, at matched token cost, against both a pre-hoc baseline and an oracle, with abstention as a first-class action.

## 8. Baselines reproduced before any novel work

| # | Baseline | Source | Success condition |
|---|---|---|---|
| B1 | maj@k, k ∈ {1,2,4,…,64}, MATH-500 | standard | monotone-then-plateau; plateau located |
| B2 | pass@k on the identical pool | 2407.21787 | coverage rises after maj@k flattens |
| B3 | PRM-weighted majority and PRM best-of-N | 2502.06703 / 2501.07301 | PRM ≥ maj at small N, margin consistent with published ~1.4% |
| B4 | Single larger-model pass (frontier anchor) | — | sanity-consistent with published figures for that model |

Hard ordering: B1–B4 complete before the predictor or search arm is built.

## 9. Literature claims independently verified

Reported as *reproduced / partially / not reproduced / out of scope*.

| # | Claim | Source | Test |
|---|---|---|---|
| V1 | A 1B model with compute-optimal TTS beats a 405B model | 2502.06703 | Check scope: MATH-500-only, N=512, PRM-dependent; the paper itself reports underperformance on AIME24. Does gap-closing survive on OlympiadBench at N=64? |
| V2 | PRMs beat majority voting | 2501.07301 | Reproduce the margin; does it clear the paired bootstrap CI? |
| V3 | Self-consistency plateaus, can decline after peak on hard items | 2508.00410 | Locate plateau-N per benchmark and difficulty band |
| V4 | PRM argmax winners are length-biased | 2606.09078 | Regress winner length and step count against pool median |
| V5 | Cheap agreement signals give large savings at ~zero accuracy cost | 2305.11860 | Reimplemented as a predictor comparator; measured here |

## 10. Headline hypotheses — all answerable at N=64

Accept conditions frozen and tagged Day 10.

| ID | Stage | Hypothesis | Accept if (at N=64, the MUST floor) |
|---|---|---|---|
| **H1** | DIAGNOSE | The allocation choice matters and is heterogeneous | Oracle allocation ≥ best fixed policy by **≥8 points** at matched tokens on ≥1 dev benchmark, paired bootstrap CI excluding 0; **and** each of A0, A1, A2, A4 is the oracle action for ≥5% of problems at ≥1 budget level |
| **H2** | PREDICT | Cheap probe evidence predicts the winning action | 4-class macro-AUROC **≥0.70**, grouped 5-fold CV, above majority-class and agreement-threshold comparators |
| **H3** | PREDICT | Post-hoc sampled evidence beats pre-hoc query text | Post-hoc macro-AUROC exceeds the pre-hoc embedding classifier by **≥0.05**, paired over problems |
| **H4** | ALLOCATE | Predicted allocation improves the quality–compute frontier | At matched tokens, beats every fixed policy on ≥3 of 5 budget levels, **and** closes **≥50%** of the random→oracle gap |

Matched-token budget levels are defined relative to the N=64 floor: B ∈ {A1-equivalent at N = 4, 8, 16, 32, 64}. An N=128 pool, if produced, adds a sixth level and strengthens plateau/crossover analysis but is not required by any accept condition.

H1 is both the first hypothesis and the premise check; gate G1 tests it in miniature on Day 4.

## 11. Everything else, correctly classified

None of the following is a headline research question.

| Item | Classification | Serves |
|---|---|---|
| PRM-argmax crossover as N grows | **Supporting analysis** (DIAGNOSE) | Validity region of the SELECT action |
| PRM length/step bias | **Supporting analysis** (mechanism) | Explains the crossover; verifies V4 |
| Selection gap (pass@k − best selector) | **Supporting analysis** (DIAGNOSE) | Upper bound on SAMPLE+SELECT |
| **N=128 pools** | **SHOULD / planned extension** | Sharpens crossover and plateau; adds a sixth budget level |
| Second PRM (Skywork-1.5B) | **Robustness** — SHOULD | Is SELECT's value verifier-dependent? |
| Verifier-free selectors | **Robustness** — SHOULD | Is a free signal as good as a 7B verifier inside A2? |
| Local 32B comparator | **SHOULD** | Context for the small-vs-large gap; does not answer H1–H4 |
| Third search budget / expansion to 200 problems | **SHOULD** | Strengthens the A3 region estimate |
| Probe size k ∈ {2,4,8} | **Supporting analysis** — SHOULD | How cheap can the evidence be? |
| Feature ablation | **Supporting analysis** — MUST | A predictor with an unidentified driver is not a result |
| Segmentation-convention ablation | **Validity check** | Are PRM scores robust to delimiter choice? |
| Extraction / truncation / ambiguity rates | **Validity checks** — MUST | Reported as first-class metrics |
| Seed / replicate stability | **Robustness** — SHOULD | Is the landscape stable across sampling replicates? |
| Model scale (smaller policy) | **Generalization** — STRETCH | Does the answer shift with policy size? |
| Thinking-mode arm | **Generalization** — STRETCH | Does the landscape survive longer reasoning? |
| Determinism / parity tests | **Validity checks** — MUST | Enable paired statistics and demo integrity |

## 12. Models, benchmarks, splits

| Role | Choice | Tier | Why |
|---|---|---|---|
| Primary policy | **Qwen3.5-4B, non-thinking**, `max_tokens=1024` | MUST | Non-thinking output (~600–900 tok) makes N=64 affordable on any backend *and* keeps matched-token comparisons unconfounded — thinking mode varies length 3–5×, which would void H1 and H4. Updated from Qwen3-4B on Day 1 (G0): neither model appears in the original baseline papers (both predate Qwen3), so the switch costs no comparability to published numbers and gains a stronger current checkpoint in the same size class and dual-mode design |
| Verifier | Qwen2.5-Math-PRM-7B (rung 1), smaller/quantized PRM (rung 2) | MUST | Three-rung ladder in §27.3a; rung 3 is a mentor decision gate, not a substitution |
| Frontier anchor | one API/Bedrock model, single pass | MUST | The assignment's "single larger model call"; the program's Bedrock expectation |
| Local 32B comparator | Qwen2.5-32B-Instruct AWQ-4bit | **SHOULD** | Token-symmetric context; not required for H1–H4 |
| Second verifier | Skywork-PRM-1.5B | SHOULD | Robustness on A2 |
| Second policy | smaller Qwen3 | STRETCH | Generalization |

**New selection constraint (from §27):** the primary policy must be available on **at least two execution backends** — local weights plus at least one hosted API — verified on Day 1 as part of gate G0. If Qwen3.5-4B is not served by an available API host, substitute the nearest small open-weight model that is, and record the substitution in `notes/`.

| Split | Content | Use |
|---|---|---|
| Dev | MATH-500 (500) + OlympiadBench slice A (300) | Reproduction, landscape, predictor fitting via grouped CV, allocation simulation, ablations |
| **Held-out** | OlympiadBench slice B (100, disjoint) + AIME25 (30) | One evaluation, Day 18 |

Slice B is in-distribution held-out; AIME25 is out-of-distribution and harder, and is **ordinal-only** (n=30). Held-out IDs are SHA-256 hashed and committed Day 2; no held-out generation before Day 18; no re-tuning after.

## 13. Pool design — N=64 floor, N=128 extension

```
pool_id = blake2s(policy_ref ‖ backend_ref ‖ benchmark_id ‖ problem_id
                  ‖ temp ‖ top_p ‖ max_tokens ‖ seed ‖ N)
```

`backend_ref` — which identifies the **backend and provider deployment**, not merely the backend type — is part of the hash: a pool generated on a different backend or a different provider deployment is a *different pool*, never a continuation.

| Pool | Problems | N | Tier |
|---|---|---|---|
| P1 MATH-500 | 500 | **64 MUST**, extend to 128 SHOULD | MUST |
| P2 OlympiadBench-A | 300 | **64 MUST**, extend to 128 SHOULD | MUST |
| P3 Hard subset (lowest pass@8 quartile) | 100 | 128–256 | SHOULD |
| P4 Held-out Olympiad-B | 100 | 64 | MUST, Day 18 |
| P5 Held-out AIME25 | 30 | 64 | MUST, Day 18 |
| P6 Replicate of P2 (second sampling replicate) | 300 | 64 | SHOULD |
| P7 Smaller policy on P1 subset | 200 | 64 | STRETCH |

**Nested prefixes.** An N=128 pool *contains* the complete N=64 experiment by truncation. So the extension is strictly additive: generate to 64 first, checkpoint, and continue to 128 only if compute allows. A shortfall never invalidates work already done.

Records store: text, three step-segmentation conventions, extracted and canonicalized answers, per-token logprobs *where the backend provides them*, cumulative logprob where available, token counts, finish reason, declared outcome, wall-clock, and the full inference-metadata block from §27.

## 14. Selectors

Machinery *inside* action A2, not a research question.

| Selector | Tier |
|---|---|
| Plain majority (defines A1) | MUST |
| Oracle pass@k (ceiling) | MUST |
| PRM-weighted majority (defines A2) | MUST |
| PRM-argmax (crossover / length-bias analysis) | MUST |
| Self-certainty weighted | SHOULD |
| Length-normalized majority | SHOULD |
| Cluster-then-vote | SHOULD |
| PRM reductions (last/min/mean/product) | SHOULD |

## 15. The bounded search arm — reduced MUST workload

| Scope | Configuration | Tier |
|---|---|---|
| **Committed minimum** | **100–150 representative problems × 2 matched-token budgets** (matched to A1 at N≈8 and N≈32), beam width 4, one policy, one PRM | **MUST** |
| Extension | third budget (N≈64-equivalent) and/or expansion to 200 problems | SHOULD |

Problems are **stratified** across both dev benchmarks and across five pass@1 difficulty bands, so 100–150 still supports the question "does A3 occupy a meaningfully distinct region of the landscape." Two budgets bracket the region where the literature expects search to win (low budget) and lose (high budget); the third budget sharpens the trend but is not required to detect the region.

**Why it stays MUST:** it is the only way to test whether the action space includes A3. Removing it does not simplify the question — it silently answers a 4-way question 3-way.

**Token accounting:** charged for policy tokens *including discarded beam branches* plus PRM forwards. Undercounting either invalidates H1 and H4.

**Excluded:** DVTS, MCTS, lookahead, budget forcing.

## 16. The predictor

**Labels:** oracle action `a*(q,B)` per problem per budget level. Primary 4-class over dev; secondary 5-class on the search subset.

**Features** — from the k=4 probe, grouped by backend dependency:

| Group | Features | Backend requirement |
|---|---|---|
| **Agreement** (core) | top-1 vote fraction, top-2 margin, normalized entropy, distinct-answer count | **none** — works on any backend |
| **Shape** | mean/variance of output length, mean step count | none |
| **Hygiene** | extraction-failure fraction, truncation fraction | none |
| **Confidence** | mean and min per-token logprob, mean self-certainty, cumulative-logprob spread | **requires logprobs** — available locally, inconsistent via API |
| *(ablated)* | mean PRM score of probe samples | requires PRM at probe time |

This grouping is deliberate: the agreement group alone must be sufficient to test H2, so a backend that does not expose logprobs degrades the feature set without invalidating the hypothesis. The feature ablation (A5, MUST) already measures exactly what the confidence group contributes, so any loss is quantified rather than unknown.

**Model:** multinomial logistic regression as primary — interpretable, and coefficient signs are themselves a finding. Gradient-boosted trees as a STRETCH capacity check.

**Protocol:** grouped 5-fold CV by problem, stratified by benchmark. Dev only. Coefficients frozen and tagged before Day 18.

**Comparators:**

| Comparator | Represents |
|---|---|
| Majority class | Trivial floor |
| Fixed agreement threshold on the probe | Adaptive-Consistency's published signal class (V5) |
| **Pre-hoc embedding classifier on question text** | The control for H3 |
| Difficulty-tier oracle (true pass@1 band) | Ceiling on difficulty alone |
| Full oracle | Ceiling |

The pre-hoc comparator is non-negotiable: without it H3 is unfalsifiable.

## 17. Experiment matrix, by stage

| # | Stage | Experiment | Configuration | Hypothesis | Tier |
|---|---|---|---|---|---|
| E0 | — | **Thin end-to-end slice** | 100 problems, N=64, full Diagnose→Predict→Allocate chain | preliminary H1/H2/H4 | **MUST, Day 11** |
| E1 | — | Baseline reproduction | P1, 4 selectors, k ∈ {1..64} | B1–B3 | MUST |
| E2 | DIAGNOSE | Action-value landscape | P1∪P2 × {A0,A1,A2} × 5 matched-token budgets × 5 difficulty bands | H1 | MUST |
| E3 | DIAGNOSE | Search arm | beam × 2 budgets × 100–150 stratified problems vs A1, A2 | H1 (is A3 in the space?) | MUST |
| E4 | DIAGNOSE | Crossover + length bias | P1∪P2 (+P3 if produced) | supporting, V4 | MUST/SHOULD |
| E5 | PREDICT | Predictor + comparators + feature ablation | 4-class, grouped 5-fold CV, 5 comparators, 5 feature sets | H2, H3 | MUST |
| E6 | PREDICT | Search-inclusive predictor | 5-class, search subset | H2 secondary | SHOULD |
| E7 | ALLOCATE | Pareto frontier | 7 policies × 5 matched-token budgets over cached pools | H4 | MUST |
| E8 | — | Larger-model comparators | frontier anchor single pass (MUST); local 32B (SHOULD) | cost-per-correct | MUST/SHOULD |
| E9 | — | Held-out evaluation | P4, P5, frozen controller, single pass | all | MUST |

**Fixed policies in E7:** Miser (A0 always), Spendthrift (A1 at max budget), Uniform-Select (A2 always), Gambler (random at matched rate), Fortune Teller (pre-hoc routing), **Detective** (post-hoc predictor), Oracle.

**E0 is new in v2** and exists to satisfy the execution rule in §29: it is a deliberately thin vertical slice that proves the whole chain works end to end before any expensive SHOULD work begins.

## 18. Metrics, mapped

| Metric | Serves |
|---|---|
| accuracy per action per budget | H1, H4 |
| oracle-allocation advantage over best fixed policy | **H1** |
| oracle-action distribution | **H1** heterogeneity clause |
| pass@k, selection_gap@k | supporting |
| crossover N\*, winner length/step delta | supporting, V4 |
| predictor macro-AUROC, per-class AUPRC, ECE, confusion matrix | **H2** |
| pre-hoc vs post-hoc AUROC delta | **H3** |
| tokens_to_matched_accuracy | **H4** |
| allocation regret, normalized to random→oracle span | **H4** |
| cost_per_correct_answer (tokens, backend-hours, USD) | program requirement |
| truncation_rate, extraction_failure_rate, ambiguity_rate | validity |
| logprob_availability_rate | validity (backend transparency) |

**Client translations** — labelled as projections, not measured business outcomes:

| Research metric | Gateway metric |
|---|---|
| accuracy at matched tokens | quality at a fixed budget |
| tokens_to_matched_accuracy | cost to reach target quality |
| A0 rate | share resolved cheaply |
| A4 rate | abstention / upstream-escalation rate |
| compute spent on A0-optimal queries under Spendthrift | unnecessary-compute rate |
| allocation regret | headroom vs perfect allocation |

## 19. Statistics

- **Pairing.** Every action, budget, and policy is evaluated over the identical frozen pool; all comparisons are per-problem paired differences. Pairing depends on the pool being *frozen*, not on being regenerable — see §27.
- **Bootstrap.** 10,000 resamples over the problem axis, BCa intervals, for every reported difference.
- **Paired binary.** McNemar for correctness flips between actions at fixed budget.
- **Multiplicity.** Holm–Bonferroni within declared families (actions; comparators; ablations).
- **Effect sizes** always beside p-values; no p-value alone.
- **Stratification.** Headline results reported overall *and* by five pass@1 bands.
- **Honesty rule.** No mean, p-value, or CI for any cell with fewer than 5 replicates; those report medians and ordinal statements. AIME25 ordinal-only regardless.
- **Backend purity.** No statistical comparison may span pools generated on different backends. Cross-backend results are reported side by side as separate rows, never pooled.

## 20. Failure taxonomy

Closed set. An uncomputable metric records `null` plus status — never a silent zero.

`ok` · `no_boxed_answer` · `extraction_ambiguous` · `equivalence_timeout` · `length_truncated` · `step_segmentation_failed` · `prm_score_missing` · `logprobs_unavailable` · `pool_incomplete` · `oom` · `search_budget_exhausted` · `beam_collapsed` · `model_load_failed` · `backend_unavailable` · `rate_limited`

Any problem with >20% non-`ok` samples is flagged and reported separately, never dropped. Unrecognized status fails the loader loudly.

## 21. Cached vs live

**Generated once, reused forever:** pools; PRM step-score arrays; nested prefixes; predictor feature tables; every policy replay; the entire demo in benchmark mode.

**Cannot be cached:** beam search (the verifier steers decoding); any new policy/temperature/max_tokens/backend; held-out pools (must not exist before Day 18); live demo mode; the frontier anchor pass.

## 22. Architecture

Twelve components. **R** supports research methodology, **O** makes the finding operational, **P** improves reproducibility/observability.

| Component | Role | Justification |
|---|---|---|
| `gateway` | FastAPI client-facing API | O |
| `budget` | Token/latency accounting; charges policy tokens, discarded beams, PRM forwards | **R** — matched-token claims are void without it |
| `generation` | **Backend-abstracted** generation; batched, resumable, config-hashed | R P |
| `backends` | Pluggable drivers: local vLLM · hosted API · Bedrock. One interface, recorded metadata | **R P** — new in v2; see §27 |
| `pools` | Content-addressed store; nested-prefix views | R P |
| `scoring` | PRM scoring; offline batch and online single | R O |
| `search` | Bounded PRM-guided beam search | R |
| `selectors` | Pure functions over (pool, scores) | R |
| `controller` | Featurize → predict → choose action | **R + O** — the finding, as code |
| `telemetry` | Structured decision records, trace IDs | P O |
| `replay` | Replays any controller over cached pools at zero inference cost | **R P O** |
| `evaluation` | Metrics, paired bootstrap, figures | R |
| `ui` | Observability + demo | O |

**The load-bearing decision:** `controller` is one object with one interface, consumed by `replay` (offline, ~800 problems, zero inference) and `gateway` (online, one request). The policy evaluated is byte-identically the policy served, enforced by `test_controller_parity.py` — **which now exists from Week 2, not Week 4.**

```python
class Probe(TypedDict):
    samples: list[Sample]
    features: dict[str, float]

class Decision(TypedDict):
    action: Literal["stop", "sample", "select", "search", "abstain"]
    budget_grant: int
    class_probs: dict[str, float]
    rationale: dict[str, float]

class Controller(Protocol):
    def featurize(self, probe: Probe) -> dict[str, float]: ...
    def decide(self, probe: Probe, budget: Budget) -> Decision: ...

class Backend(Protocol):
    def generate(self, prompts: list[str], cfg: DecodeConfig) -> list[Sample]: ...
    def capabilities(self) -> BackendCaps: ...   # logprobs? seeds? max concurrency?
```

Neither `replay` nor `gateway` may contain allocation logic of its own.

**API contract** — unchanged from v1:

```
POST /v1/solve
{ "query": str,
  "budget": {"max_tokens": int, "max_latency_ms": int | null},
  "policy": "detective" | "spendthrift" | "uniform_select" | "prehoc" | "random" | "miser",
  "trace": bool }

200 →
{ "outcome": "answered" | "escalated" | "declined",
  "answer": str | null,
  "action": "stop" | "sample" | "select" | "search" | "abstain",
  "evidence": {"top1_fraction": float, "entropy": float, "class_probs": {...}},
  "spend": {"policy_tokens": int, "prm_forwards": int, "discarded_beam_tokens": int,
            "latency_ms": int, "usd_equivalent": float},
  "decision_path": [ {"stage": str, "action": str, "granted_tokens": int} ],
  "trace_id": str }
```

`declined` returns `answer: null` plus a machine-readable reason. Budget exhaustion mid-escalation returns the best answer so far, flagged `budget_exhausted` — anytime by construction.

## 23. Repository

```
marginal-token/
├── README.md                    # reproduce from this alone
├── Makefile                     # reproduce-headline | verify-determinism | demo | test
├── pyproject.toml / uv.lock
├── configs/
│   ├── policies/ prms/ backends/ benchmarks/ pools/ experiments/
├── src/marginal_token/
│   ├── gateway/ budget/ generation/ backends/ pools/ scoring/ search/
│   ├── selectors/ controller/ replay/ evaluation/ telemetry/ answers/
├── tests/
│   ├── test_determinism.py           # local backend: same config → identical bytes
│   ├── test_controller_parity.py     # offline decision == online decision  (Week 2)
│   ├── test_answer_equivalence.py    # 200 hand-checked pairs
│   ├── test_budget_accounting.py     # discarded beams + PRM forwards charged
│   ├── test_backend_metadata.py      # every sample carries full provenance
│   └── test_taxonomy.py              # unknown status fails loudly
├── ui/                          # static demo, reads cached artifacts
├── notes/                       # dated, append-only decision + experiment log
├── report/                      # incrementally written report sections
└── results/                     # figures, tables, manifest.json
```

## 24. Reproducibility from the README alone

1. `uv sync` against a committed lockfile; model revisions pinned by HF commit SHA or API model ID.
2. `make verify-determinism` — on the **local-weights backend**, regenerates a small pool and asserts byte-identity against a committed hash. On API backends this reports `determinism: not guaranteed by backend` and the frozen pool artifact plus its hash becomes the reproducibility unit instead.
3. `make reproduce-headline` — rebuilds every headline table and figure **from committed cached artifacts, with no GPU and no API key required.** This is the critical property and it is backend-independent.
4. `make reproduce-pools` — regenerates from scratch; requires a backend; documented cost and duration per backend.
5. `results/manifest.json` maps every figure and table to pool hashes, backend refs, and code commit.
6. `notes/` is append-only and dated.

**Hard completion criterion:** a clean clone must reproduce the headline tables. This is not negotiable and is checked on Day 20.

## 25. Real-life use case — Compute-Aware Reasoning Gateway

**Scope: a well-engineered single-service artifact, not a platform.** One FastAPI service, one policy model, one PRM, a static demo page. No autoscaling, no multi-tenancy, no orchestration.

**Scenario.** A team runs a small reasoning model on their own infrastructure because requests must stay local. Today they choose between always spending a large uniform sampling budget (wasted on the easy majority) or a small one (silent failures on the hard minority). Neither knows when the model cannot recover an answer at all.

The gateway adds the missing behaviour: probe cheaply, spend only where evidence says spending pays, choose *which kind* of spending, and **decline** when evidence says nothing will help — routing upstream deliberately.

**Honesty boundary, in the report:** all evidence is math-benchmark evidence. The gateway framing is a labelled projection onto a deployment shape, not a measured business outcome.

## 26. Demo

**Benchmark mode** (default; cached, instant, zero inference cost, has ground truth):

1. Audience picks a problem and a token budget.
2. **Probe panel** — four samples stream; answers appear as bars that grow and merge.
3. **Evidence panel** — agreement fraction, entropy, distinct-answer count update live.
4. **Controller panel** — class probabilities across actions; the chosen action lights up.
5. **Spend panel** — token, PRM-forward, and latency meters run as the grant is consumed.
6. **Outcome** — answer with full vote distribution, or refusal with reason. Ground truth revealed after.
7. **Comparison mode** — same problem, same budget, seven policies side by side, meters draining in parallel. The moment that lands: Spendthrift burns its whole budget and is still wrong while Detective declines after four samples and spends almost nothing.

**Live mode** — audience-submitted question, identical components, real inference on whatever backend is available, no ground truth; the UI shows the decision process and says so.

Both modes call the same `Controller.decide`; `test_controller_parity.py` enforces it.

## 27. Compute plan — resource-agnostic

**Principle.** The research question specifies an *inference workload*, not a machine. Compute backends are replaceable execution resources. A remote RTX 3090 over SSH is treated as an **opportunistic accelerator**, not a project dependency.

### 27.1 Backend abstraction and provenance

Every generated sample records:

```json
{"model_id": "...", "backend": "local_vllm | api_host | bedrock",
 "provider": "...", "revision_or_api_model": "...",
 "temperature": 0.8, "top_p": 0.95, "max_tokens": 1024,
 "seed": 1234, "seed_honored": true|false|"unknown",
 "logprobs_available": true|false, "quantization": "bf16|awq4|...",
 "generated_at": "...", "pool_id": "..."}
```

`test_backend_metadata.py` fails any sample missing this block.

**Two hard rules:**
1. **One backend/provider deployment per pool, and one *condition* per pool.** `backend_ref` is inside the pool hash. If a backend becomes unavailable mid-generation, the partial pool is either completed on the same deployment later, or closed at whatever N was reached and reported at that N. It is never continued on a different backend or provider deployment, and never continued with a materially different model or decode configuration — see the compatibility contract in §27.6.
2. **No statistical comparison spans backends.** Cross-backend results appear as separate rows, never pooled.

### 27.2 Determinism vs pairing — the resolution

| Property | Depends on | Status |
|---|---|---|
| **Paired evaluation** (H1–H4) | The pool being **frozen** | **Preserved on every backend.** All selectors, budgets, and policies read one identical cached artifact. |
| **Byte-level regeneration** | The backend honoring seeds | **Local-weights backend only.** Verified by G4 when that backend is in use; recorded as `not guaranteed` otherwise. |

This is the one place v2 relaxes a v1 MUST, and it is the correct relaxation: the frozen pool artifact plus its hash is the reproducibility unit, and `make reproduce-headline` works from it with no backend at all.

### 27.3 Workload ledger

Volumes assume ~800 output tokens per sample and ~150 input tokens per prompt.

| Workload | Token volume | 3090-class GPU-h | Needs GPU-class compute? | API-capable? | Approx. API cost | Fallback if preferred resource unavailable |
|---|---|---|---|---|---|---|
| **P1+P2 pools, N=64** (MUST) | ~41 M out, ~8 M in | **8–14 h** | preferred | **Yes** | **~$2–15** (verify host pricing Day 1) | Generate via API host; if budget-constrained, reduce OlympiadBench slice A from 300 → 200 problems |
| P1+P2 extension to N=128 (SHOULD) | +41 M out | +8–14 h | preferred | Yes | +$2–15 | Skip. Nested prefixes mean N=64 work is already complete |
| **PRM scoring, P1+P2** (MUST) | ~41 M prefill per pass | **4–10 h** (7B PRM) | **Yes — no API exists** | **No** | — | **PRM ladder (§27.3a):** rung 1 primary 7B PRM on an available GPU → rung 2 smaller/quantized PRM on any available compatible GPU environment (~2–5 h) → rung 3 **mentor decision gate G10**, not an automatic substitution |
| **Search arm, 100–150 × 2 budgets** (MUST) | ~3–5 M out + PRM forwards | **4–7 h** | preferred | Partially — via stop-sequence stepping | ~$1–4 policy tokens + GPU-side PRM | Hybrid: policy generation via a compatible API backend, PRM scoring at ladder rung 1 or 2. Reduce to 100 problems × 2 budgets |
| **Held-out P4+P5, N=64** (MUST) | ~6.7 M out | **1.5–3 h** | preferred | Yes | ~$0.5–3 | Same backend as dev pools if at all possible; if not, report as a separate backend row with the caveat stated |
| **Frontier anchor, single pass** (MUST) | ~0.5 M out | — | No | **Yes** | **~$3–8** (Bedrock or API) | Reduce to 200 dev + 130 held-out; cap spend |
| Local 32B comparator (SHOULD) | ~0.9 M out | 2–3 h | No | Yes | **~$0.20–1** via API | Run via API instead of locally — cheaper and simpler than the 4-bit local route |
| Pre-hoc embedding comparator (MUST) | ~930 embeddings | negligible | No | Yes | ~$0–1 | Local sentence-transformer at $0 |
| P3 / P6 / P7 (SHOULD/STRETCH) | varies | 10–20 h | preferred | Yes | ~$3–10 | Skip entirely |
| **All analysis, replay, ablations, demo** | 0 | **0** | **No** | n/a | **$0** | None needed — this is the point of the frozen-pool design |

**MUST totals:** ~52 M output tokens of generation plus ~41 M tokens of PRM prefill. Executed entirely on a 3090-class GPU: **~18–34 GPU-hours.** Executed hybrid — policy generation via a compatible API backend, PRM scoring at ladder rung 2 on whatever compatible GPU environment is available: **~$6–25 in API spend plus ~6–15 hours of modest GPU time.** Both paths fit the stated budgets.

The N=64 floor is what makes this true. At N=128 the generation half roughly doubles; that is exactly why it is SHOULD.

### 27.3a The PRM resource ladder

Action A2 (SELECT) is research-critical and depends on a step-level process reward model. PRMs emit per-step scalar rewards rather than chat completions, so **no hosted API provides them.** The ladder below is a *resource* ladder — each rung runs the same class of verifier at a different cost — and it deliberately stops before becoming a *method* ladder.

| Rung | Configuration | Cost | Status |
|---|---|---|---|
| **1 — Preferred** | Primary 7B PRM (Qwen2.5-Math-PRM-7B) on whatever remote or local GPU-class environment is available | ~4–10 h per full scoring pass | Default |
| **2 — Fallback** | Smaller and/or quantized PRM (e.g. a 1.5B PRM in 4-bit, ~1 GB) on **any available compatible GPU environment** | ~2–5 h per pass | Automatic; requires only a `notes/` entry recording the rung and the environment |
| **3 — Neither available** | — | — | **Triggers gate G10: mentor decision. No automatic substitution.** |

**Rung 2 names no specific provider or platform on purpose.** A hosted notebook GPU, an institutional machine, a borrowed card, or a short paid cloud-GPU rental all qualify equally. None of them is an infrastructure assumption; the requirement is "a compatible GPU environment," and any one that satisfies it is acceptable. Whichever is used is recorded in the pool provenance block.

**Why rung 3 is a gate and not a fallback.** The available substitutes for a step-level PRM — a whole-solution outcome reward model, or an LLM-as-judge critic via API — are *scientifically different verifiers*. Swapping either in would redefine what A2 is, which would change the meaning of H1 (the action-value landscape), H2 (the labels the predictor learns), and H4 (the frontier). Reporting that as "A2" would misrepresent the experiment. So rung 3 escalates rather than degrades.

Worth noting when G10 fires: a few hours of a rented cloud GPU typically costs single-digit dollars and sits comfortably inside the $50 AWS allowance, so "secure a compatible GPU environment" is usually both the cheapest and the scientifically cleanest resolution. That should be considered before any redefinition of A2.

### 27.4 Preferred allocation of scarce GPU time

If remote 3090 access is available, spend it in this order — highest value first:

1. **PRM scoring.** No API alternative exists; this is where GPU access is irreplaceable.
2. **Search arm.** Many short interleaved generations; API round-trip latency hurts most here.
3. **P1+P2 pool generation.** Valuable but fully API-substitutable.
4. **N=128 extension**, then SHOULD pools. First to be dropped.

Two distinct shortfall scenarios, with different consequences:

**Scenario A — no dedicated remote GPU, but some compatible GPU environment is available.** Policy generation and the frontier anchor go to compatible API backends; PRM scoring runs at ladder rung 2 on whatever GPU environment is at hand; the search arm runs hybrid at the 100-problem floor; the 32B comparator runs via API. **No hypothesis is lost.** Scale degrades — OlympiadBench slice A may drop to 200 problems, N stays 64, the SHOULD tier is skipped — and the report states the executed configuration plainly.

**Scenario B — no GPU-class environment available at all.** This is *not* a scale problem. Action A2 (SELECT) is research-critical and has no API path, so it cannot be executed as designed. This triggers **gate G10** rather than an automatic fallback, because the available substitutes are scientifically different verifiers, not cheaper versions of the same one. Do not proceed silently.

### 27.5 Budget

| Item | Planned | Ceiling |
|---|---|---|
| Frontier anchor (Bedrock or API) | $3–8 | $10 |
| S3 static demo hosting + artifacts | $3 | $5 |
| Embeddings | $0–1 | $2 |
| **API generation reserve** (used only if GPU unavailable) | $0 | **$25** |
| **Total planned if GPU available** | **~$8** | |
| **Total if GPU unavailable throughout** | **~$20–35** | |

AWS budget alarms at $10 / $25 / $40 configured Day 1. API spend tracked in `notes/` per run. **Neither AWS nor any API host is forced into the project where it adds nothing** — the frontier anchor is the only element genuinely requiring a hosted model, and it is required by the assignment rather than by convenience.

### 27.6 Policy-generation fidelity and the pool compatibility contract

If policy generation moves to a hosted API backend, the primary model and decode configuration are preserved **as closely as the provider allows**, and every deviation is recorded. A backend change is an execution detail; a model or configuration change is a *condition* change.

**Compatibility contract — these must be identical for samples to belong to one pool:**

| Attribute | Rationale |
|---|---|
| Model family, size, and weights revision | Different weights sample from a different distribution, which is exactly what H1 measures |
| Quantization / numeric precision | Changes the sampling distribution and therefore coverage and pass@k |
| Temperature, top-p, top-k, repetition penalty | Directly determines pool diversity |
| `max_tokens` | Load-bearing: matched-token accounting is confounded if the effective output-length distribution shifts |
| Prompt template, system prompt, stop sequences | Changes the task as presented to the model |

**Also part of pool identity:** the **backend / provider deployment** itself. A single frozen pool uses **one model revision, one decoding configuration, and one backend/provider deployment.** If execution moves to another provider or backend, create a new pool and a new condition rather than continuing the existing one — **even if the nominal model weights are identical.** Providers differ in serving stack, kernel and attention implementation, batching, and clamping behaviour in ways that are not always documented, so treating the deployment as part of the condition is what makes provenance, paired evaluation, and reproducibility defensible without having to prove those differences absent.

**Recorded but permitted to differ within one pool:** request routing and retries inside a single deployment, seed value and whether it was honoured, logprob availability, latency, timestamps.

**Operational consequences:**

1. **Never mix materially different policy models or providers inside one pool and treat them as one condition.** If a provider serves only a different quantization of the same weights, that is a *different* pool and a *different* condition — reported as a separate row, never merged.
2. **Provider-side clamping counts as material** if it changes the output-length distribution, even when the weights are identical, because matched-token comparisons depend on that distribution. Verify `max_tokens` and any provider truncation behaviour on Day 1 as part of gate G0.
3. If no available API host serves the primary model within the contract, the correct move is to **change the primary model for the whole project** (recorded in `notes/`, with the Day-1 baseline re-run) rather than to run some pools on one model and some on another.
4. A pool validator enforces the contract in code; `test_backend_metadata.py` fails any sample whose provenance block is incomplete or whose contract fields disagree with the pool manifest.

## 28. Risks and fallbacks

| # | Risk | P×I | Mitigation | Fallback |
|---|---|---|---|---|
| R1 | Extraction/equivalence silently wrong → all numbers invalid | **9** | Day 3 dedicated; 200-pair hand-checked golden set; failure rate is a headline metric | Restrict to unambiguous numeric answers, report the restriction |
| R2 | PRM segmentation wrong → PRM scores are noise | **9** | Gate G3 Day 5 before any scale scoring; three conventions cached | Try the other segmentation conventions, then ladder rung 2. Replacing the PRM with a whole-solution ORM would redefine A2, so it goes through **G10**, not through the fallback chain |
| R0 | **Compute backend unavailable or intermittent** | **8** | Backend abstraction from Day 6; G0 verifies two backends and provider clamping on Day 1; N=64 floor | §27.4 Scenario A: hybrid execution, API generation reserve, slice A reduced to 200 problems. Scenario B (no GPU-class environment at all) → **G10** |
| R10 | **No compatible GPU environment for PRM scoring** | **6** | PRM ladder §27.3a; rung 2 accepts any compatible GPU environment | **G10 mentor decision.** A short paid cloud-GPU rental is usually the cheapest and cleanest resolution and fits the budget |
| R11 | No API host serves the primary model within the compatibility contract | **4** | G0 Day 1 verifies model availability and provider clamping on ≥2 backends | Change the primary model for the **whole** project and re-run the Day-4 baseline; never split pools across models |
| R3 | Losing 2–3 working days | **7** | E0 thin slice on Day 11 proves the chain early | Cut order in §31 |
| R4 | H1 fails — no allocation headroom | **7** | Gate G1 Day 4, before the brief is presented | Promote OlympiadBench to primary; then a weaker policy; then mentor escalation |
| R5 | Search arm adaptation overruns | **6** | Gate G5 Day 12; adapt the published repo; committed floor is only 100 problems × 2 budgets | Drop to 100 × 2; if broken Day 14, drop A3 → 3-way action space, reported as a negative result |
| R6 | Predictor AUROC < 0.60 | **5** | Feature ablation designed in advance so failure is diagnosed | Report as headline negative result; narrative shifts to H1's landscape with oracle bounds |
| R7 | Backend does not expose logprobs | **4** | Feature groups separated in §16 so agreement features alone can test H2 | Drop the confidence group; the MUST feature ablation quantifies exactly what was lost |
| R8 | Pool determinism unverifiable on the available backend | **3** | §27.2 resolution: frozen artifact is the reproducibility unit | Record `determinism: not guaranteed by backend`; `make reproduce-headline` is unaffected |
| R9 | Demo not ready | **2** | Built Day 17 from cached artifacts; gateway skeleton exists from Week 2 | Benchmark mode only; drop live mode |

## 29. Execution rules

**Rule 1 — MUST before SHOULD.** No SHOULD experiment may consume significant inference or implementation time until **E0** has successfully produced end-to-end Diagnose → Predict → Allocate results on the 100-problem thin slice. SHOULD analyses that run free over already-cached artifacts (verifier-free selectors, PRM reductions, probe-size ablation) may proceed earlier; expensive SHOULD *generation* and *scoring* (N=128 extension, P3, P6, P7, second PRM, third search budget, local 32B) waits.

**Rule 2 — N=64 first, always.** Generation proceeds to N=64 across all MUST pools before any pool is extended toward 128.

**Rule 3 — one backend per pool.** See §27.1.

**Rule 4 — no experiments after Day 17.** Gate G8.

## 30. Program deliverables — explicit

| Deliverable | Where | When |
|---|---|---|
| Project brief presented to mentors | this document | Day 5 |
| **Dated `notes/` experiment + decision log**, maintained throughout | `notes/`, append-only | Days 1–20, daily |
| **Negative results and failed hypotheses logged**, not only successes | `notes/` + report §Negative Results | continuous |
| **Mentor notification for any scope shift outside gates G0–G10** | `notes/` entry + message to mentors | as triggered |
| Report: brief summary | `report/01-summary.md` | Day 19 |
| Report: solution explanation + paper references | `report/02-solution.md` | drafted W1–W2 |
| Report: results with tables and diagrams | `report/04-results.md` | inserted W3, finalized W4 |
| **Report: What Went Well / What Went Wrong** | `report/06-retrospective.md` | Day 19 |
| **Report: Next Steps** | `report/07-next-steps.md` | Day 19 |
| **Report: literature-claim verification table (V1–V5)** | `report/03-verification.md` | drafted W1, finalized W4 |
| **Project Start vs Project End comparison** | `report/08-start-vs-end.md` | Day 19 |
| **Planned vs actual timeline, with explanation of major deviations** | `report/08-start-vs-end.md`, table against §32 hour estimates | Day 19 |
| **Interactive presentation / demo** | `ui/` + rehearsed walkthrough | Days 17, 20 |
| Verified-vs-speculative separation | `report/05-discussion.md` | Day 19 |
| **README clean-clone reproduction** — hard completion criterion | `README.md`, verified from a fresh clone | Day 20 |

## 31. Incremental report schedule

The report is never written from scratch. By the end of each week, `report/` contains:

| Week | Report content in report-ready form |
|---|---|
| **W1** | Literature review, motivation, research question, prior work and gap, baseline protocol, V1–V5 verification table skeleton |
| **W2** | Methodology, action-space definition, architecture, datasets and splits, validity protocol, experiment and ablation design, statistical plan |
| **W3** | Dev results, figures, failure analysis, preliminary discussion — inserted as each experiment finishes, not batched |
| **W4** | Held-out results, final discussion, limitations, What Went Well / What Went Wrong, Next Steps, Start vs End comparison, conclusions |

## 32. Twenty-working-day plan

**M** = MUST, **S** = SHOULD. "Background" = runs unattended on whichever backend is available. Report increments are folded into existing hours, not added.

### Week 1 — Literature, verification, baseline, brief

**Day 1 · Novelty verification + backend survey · 7 h · M**
Resolve in full text: 2604.17433, 2606.09078, 2606.08098, 2607.08065, 2506.12721. Fresh arXiv search for duplicates of the per-query-action-prediction framing. Read the compute-optimal-TTS repo; locate the beam-search entry point for Day 12. **G0:** confirm the primary policy is available on ≥2 backends; record API pricing and whether logprobs and seeds are exposed. Configure AWS budget alarms.
*Output:* novelty memo; V1–V5 skeleton; **backend capability table**; alarms live.
*Report increment:* `report/02-solution.md` references section started.
*Done when:* each paper has a written "what remains open" line, and two viable backends are named with costs.
*Decision:* **G0** — primary policy and backend set. Abandon the framing on novelty grounds (available only today).

**Day 2 · Literature review, held-out freeze, env · 7 h · M**
Write the literature review and prior-work table. Select and SHA-256 hash held-out sets from a seeded shuffle; commit. Begin `uv` project; pin deps, model revisions, API model IDs.
*Output:* `configs/benchmarks/heldout-*.yaml` with committed hashes.
*Report increment:* **W1 sections drafted** — literature, motivation, RQ, prior work.
*Done when:* the held-out hash is in git and printed in this brief.

**Day 3 · Answer extraction and equivalence · 8 h · M · highest-risk day**
Build `answers/`: extraction, canonicalization, equivalence via `math_verify`. Hand-check 200 (prediction, gold) pairs across both benchmarks as a golden test set. Record extraction-failure and ambiguity rates. **G2:** verify the chosen backend generates end to end.
*Output:* `answers/` with passing golden test; measured rates; G2 logged.
*Done when:* 200 pairs pass and rates are in `notes/`.

**Day 4 · Baseline reproduction and G1 · 7 h · M · most decisive day**
Generate 100 MATH-500 problems, N=64. Compute maj@k, pass@k, and the **G1 allocation-headroom probe** over {A0, A1, A2}. Compare maj@k against published values (B1, B2).
*Depends on:* Day 3 — extraction must be trustworthy first.
*Output:* first headroom number with bootstrap CI; G1 decision; B1/B2 partial reproduction.
*Report increment:* baseline protocol section complete.
*Done when:* G1 is decided in writing and the primary benchmark is fixed.
*Decision:* **G1.**

**Day 5 · PRM integration, G3, brief presented · 7 h · M**
Load the primary PRM; implement three segmentation conventions; score the Day-4 pool. **G3.** Reproduce B3. Finalize and present the brief.
*Output:* `scoring/` smoke-tested; G3 decision; **brief delivered to mentors**.
*Done when:* PRM AUROC >0.6 for correctness and the brief is presented.
*Decision:* **G3** — PRM rung and segmentation convention fixed.

### Week 2 — Core implementation, early integration, freeze

**Day 6 · Backend abstraction, generation engine, pool store · 8 h · M**
`backends/`: driver interface, capability reporting, provenance metadata, `test_backend_metadata.py`. `generation/`: config-hashed sweeps, resumable checkpointing, JSONL schema, outcome taxonomy. `pools/`: content-addressed store, nested-prefix views.
*Output:* `make generate CONFIG=…` runs, resumes after `kill -9`, and records full provenance.
*Done when:* a killed run resumes to a complete pool and every sample carries metadata.
*Background:* **P1 generation to N=64 begins.**

**Day 7 · Scoring engine and budget manager · 8 h · M**
`scoring/`: offline batch PRM in a separate process, sequential model loading, per-step arrays. `budget/`: exact accounting for policy tokens, PRM forwards, discarded beam tokens; `test_budget_accounting.py`.
*Done when:* accounting test passes including a synthetic discarded-beam case.
*Background:* P1 continues.

**Day 8 · Selectors, oracle, statistics · 7 h · M**
`selectors/`: 4 MUST + free variants as pure functions; oracle pass@k. `evaluation/`: paired bootstrap (10k, BCa), McNemar, Holm–Bonferroni, difficulty banding.
*Done when:* all selectors run over a real pool and the bootstrap reproduces a known synthetic CI.
*Background:* P1 finishes at N=64; **P2 begins.**

**Day 9 · Determinism, telemetry, end-to-end smoke, G4 · 7 h · M**
`test_determinism.py` → **G4** (on the local-weights backend if in use; otherwise record `not guaranteed` and rely on artifact hashing per §27.2). `telemetry/`: decision records, trace IDs. Full smoke on 50 problems: generate → score → select → metrics → figure.
*Output:* G4 outcome recorded; one complete pipeline-produced figure.
*Report increment:* methodology + architecture drafted.
*Background:* P2 continues. *Decision:* **G4.**

**Day 10 · Controller, replay, gateway skeleton, parity test, FREEZE · 8 h · M**
`controller/`: featurize, action-label computation, logistic predictor scaffold, seven policies. `replay/`: replay any controller over cached pools. **`gateway/`: minimal FastAPI `/solve` skeleton calling the same Controller — no UI, no full live inference path.** `test_controller_parity.py` passing.
**Freeze:** commit and tag `design-frozen` with hypotheses, accept conditions, E0–E9, classifications, metrics, statistical plan.
*Output:* tagged commit; **parity test green in Week 2**; frozen design doc.
*Report increment:* **W2 sections complete** — methodology, architecture, datasets, validity protocol, experiment design.
*Done when:* `replay` and `gateway` demonstrably share one Controller implementation and the tag exists.
*Background:* P2 finishes at N=64.
*Decision:* **final design freeze** — no hypothesis or accept-condition changes after today.

### Week 3 — Experiments, ablations, failure analysis

**Day 11 · E0 thin end-to-end slice · 7 h · M · unblocks SHOULD**
Score the 100-problem slice; compute action labels; run the **full Diagnose → Predict → Allocate chain** and produce preliminary H1/H2/H4 numbers. Then PRM-score all of P1+P2.
*Output:* E0 results; **Rule 1 satisfied — expensive SHOULD work unblocked from here.**
*Report increment:* preliminary results inserted.
*Done when:* one figure exists for each of H1, H2, H4 at slice scale.
*Background:* full PRM scoring; then N=128 extension **only if** on schedule (S).

**Day 12 · DIAGNOSE: landscape + search arm, G5 · 8 h · M**
Run **E2**: landscape across actions × 5 budgets × 5 difficulty bands on P1∪P2 at N=64. Adapt beam search; wire into `budget/` so discarded beams are charged. **G5** on 20 problems.
*Output:* landscape figures; oracle-action distribution; H1 partial verdict; G5 decision.
*Background:* **E3** search budget 1 of 2 on the stratified 100–150 problems.
*Decision:* **G5** — scope of A3.

**Day 13 · DIAGNOSE: supporting analyses · 7 h · M/S**
**E4:** crossover N\* with CI at N=64 (extended if N=128 exists); length and step-count regression (V4). Inspect 30 traces where PRM-argmax picked wrong and majority was right. Complete the **H1** verdict including A3.
*Output:* crossover figure; length-bias table; failure notes; **H1 final verdict.**
*Report increment:* Diagnose results + failure analysis inserted.
*Background:* E3 search budget 2 of 2.

**Day 14 · PREDICT: predictor and comparators, G6 · 8 h · M**
**E5:** fit the 4-class multinomial predictor; grouped 5-fold CV; feature ablation (including the confidence-group contribution per §16); probe-size ablation (S). All five comparators including the **pre-hoc embedding classifier**. Evaluate **H2** and **H3**. **E6** 5-class on the search subset (S). **G6.**
*Output:* AUROC table, confusion matrices, calibration curves, feature-importance figure; H2 and H3 verdicts.
*Report increment:* Predict results inserted.
*Background:* frontier anchor calls on the dev subset; 32B comparator via API (S).
*Decision:* **G6.**

**Day 15 · ALLOCATE: Pareto frontier, G7 · 7 h · M**
**E7:** seven policies × five matched-token budgets over cached pools. Evaluate **H4**. **E8:** cost-per-correct across tokens, backend-hours, USD. Free robustness runs: verifier-free selectors, PRM reductions, segmentation ablation.  **G7** resource check.
*Output:* Pareto frontier; policy comparison table with real numbers; **H4 verdict.**
*Report increment:* Allocate results + preliminary discussion inserted.
*Done when:* all four headline hypotheses have written verdicts.
*Decision:* **G7** — cut the remaining SHOULD tier if over resource budget.

### Week 4 — Held-out, delivery

**Day 16 · Gateway final integration and polish · 7 h · M**
Complete the §22 contract on the Week-2 skeleton: three outcomes, anytime budget exhaustion, live-mode inference path, telemetry wiring. Re-verify parity.
*Depends on:* Day 14 frozen coefficients; Day 10 skeleton.
*Done when:* a `declined` response is produced end to end with a machine-readable reason and parity is green.

**Day 17 · Demo build, G8 freeze · 8 h · M/S**
`ui/`: probe, evidence, controller, spend, outcome panels; comparison mode (S). Benchmark mode reads cached artifacts — zero inference cost. **G8: hard experiment freeze.**
*Done when:* an audience member can pick a problem and watch the full decision path to an outcome.
*Decision:* **G8.**

**Day 18 · Held-out evaluation, single pass · 6 h · M**
Generate P4 and P5 at N=64 on the same backend as dev if at all possible; score; run the **frozen** controller and selectors. One pass, no tuning. Frontier anchor on held-out. Final cost accounting.
*Output:* held-out table; anchor numbers.
*Done when:* held-out numbers are recorded, whatever they say.
*Decision:* none — re-tuning is explicitly forbidden today.

**Day 19 · Report finalization · 8 h · M**
Finalize held-out results, discussion, limitations. Write **What Went Well / What Went Wrong**, **Next Steps**, **Project Start vs Project End**, and the **planned-vs-actual timeline table** against §32's hour estimates with explanations for major deviations. Complete the V1–V5 verification table. Verified-vs-speculative separation; all client numbers labelled as projections.
*Note:* W1–W3 increments mean this is finalization, not first drafting.
*Done when:* every hypothesis and V-claim has a verdict and both retrospective tables are filled.

**Day 20 · Repository, reproducibility, rehearsal · 7 h · M**
README written for a stranger with no GPU and no API key. `make reproduce-headline` **verified from a clean clone** — hard completion criterion. `results/manifest.json`. Demo rehearsed twice against real components. Interactive presentation assembled.
*Done when:* a clean clone reproduces the headline tables and the demo has run start-to-finish twice.

**Total:** ~146 working hours over 20 days (~7.3 h/day), unchanged from v1. Background generation runs outside these hours.

## 33. Gates

| Gate | Day | Test | Pass | Fail action |
|---|---|---|---|---|
| **G0** | 1 | Primary policy available on ≥2 backends; pricing, logprob and seed support recorded | ≥2 backends viable | Substitute the nearest small open-weight model that is served on ≥2 backends; record in `notes/` |
| **G1** | 4 | Allocation headroom: oracle over {A0,A1,A2} minus best fixed policy, 100 MATH-500 problems, N=64 | ≥8 pts | 4–8 pts → promote OlympiadBench to primary. <4 pts on both → weaker policy. Still <4 → mentor escalation with a written pivot memo |
| **G2** | 3 | Chosen backend generates end to end | ≤4 h | Switch to the alternate backend from G0 |
| **G3** | 5 | PRM step scores predict correctness, 10 problems | AUROC >0.6 | Try the other predefined segmentation conventions → next compatible PRM rung (§27.3a) → if no valid PRM configuration remains, trigger **G10** mentor decision. Any ORM or LLM-judge substitution exists only behind G10 |
| **G4** | 9 | Determinism on the local-weights backend | pass, or `not guaranteed` recorded | If the backend cannot guarantee it, the frozen artifact plus hash becomes the reproducibility unit (§27.2). `make reproduce-headline` unaffected |
| **G5** | 12 | Beam search sane, 20 problems | within 10 pts of A1 at comparable tokens | Hold at the 100-problem × 2-budget floor; broken by Day 14 → drop A3, 3-way action space, report as negative engineering result |
| **G6** | 14 | Predictor macro-AUROC, grouped CV | ≥0.70 accept H2; 0.60–0.70 weak; <0.60 reject | Reject H2, promote negative result #2, narrative shifts to H1 + oracle bounds |
| **G7** | 15 | Resource consumption vs plan (GPU-hours and/or API spend) | ≤70% of ceiling | Cut the remaining SHOULD tier |
| **G8** | 17 | Hard experiment freeze | — | No new experiments after Day 17 |
| **G9** | 11 | **E0 thin slice produced H1/H2/H4 numbers** | pass | Expensive SHOULD work stays blocked until it does; if E0 fails by Day 13, cut all SHOULD and treat the remaining days as MUST-only recovery |
| **G10** | as triggered | **PRM ladder exhausted** — neither rung 1 nor rung 2 has a compatible GPU environment available | n/a — this gate only fires on failure | **Mentor decision, escalated the same day with a written memo.** A2 is research-critical and its substitutes are scientifically different verifiers, so it is never replaced silently. Options for the mentor, in the order I would propose them: **(a)** secure any compatible GPU environment, including a short paid cloud-GPU rental (single-digit dollars, fits the budget) — preferred, preserves the design intact; **(b)** redefine A2 around a different verifier class (outcome reward model or LLM-judge), with the change stated explicitly in the research question, hypotheses, labels, and report, and the affected results relabelled; **(c)** drop A2 and reframe the action space as STOP / SAMPLE / SEARCH / ABSTAIN, reporting the reduction as a scope change with mentor sign-off. Whichever is chosen is recorded in `notes/` with the date and rationale |

## 34. Meaningful negative results

Pre-committed; each is reportable and each is logged in `notes/` whether or not it reaches the report's headline.

1. **No allocation headroom (H1 fails).** Test-time compute is closer to its ceiling than assumed for current small models.
2. **Cheap evidence cannot predict the winning action (H2 fails).** A real limit on the adaptive-compute literature, strengthened by the feature ablation showing *which* signals fail.
3. **Pre-hoc query text matches post-hoc evidence (H3 fails).** The cheaper signal class wins — a clean joint finding with parallel routing work.
4. **Search never wins at matched tokens.** The action space is 3-way plus abstain.
5. **PRM-argmax never crosses under plain majority up to N=64.** Contradicts the reward-hacking narrative for small policies.
6. **All selectors land within noise, far below oracle.** The bottleneck is selection *information*, not the aggregation function.

## 35. Out of scope

MCTS, DVTS, lookahead. Budget forcing beyond the STRETCH verification. PRM training, fine-tuning, or distillation. Mid-generation forking or KV-cache surgery. Multilingual evaluation. Code, SQL, tool use, or any second domain. Multi-agent debate. Training a router. Production deployment, autoscaling, multi-tenancy, auth, VPC work. Distributed training. Any research claim about hardware efficiency — cost and latency are reporting axes only. Real or private data. Business or ROI claims beyond labelled projections.

## 36. Cut order if 2–3 working days are lost

1. **Remaining SHOULD tier** — N=128 extension, second PRM, P3, P6, P7, third search budget, 5-class search predictor, local 32B. Saves ~3 days. All four headline hypotheses survive.
2. **Reduce dev scale** — OlympiadBench slice A from 300 → 200 problems, search arm held at 100 problems × 2 budgets. Saves ~1.5 days. H1's A3 clause becomes descriptive; crossover may fall outside range, in which case "no crossover observed up to N=64" is the honest report.

**Cannot be cut without changing the research question:** frozen pools with paired nested prefixes; the four MUST selectors including oracle pass@k; the action-space definition and oracle-action labels; the predictor with its pre-hoc comparator; random and oracle bounds in E7; exact token accounting; the frozen held-out set run once; at least one live search budget point; controller parity; the gateway.

## 37. Thesis continuation

1. **Action prediction from hidden states** — predict the winning action from activations before any sampling. Removes the probe cost; converts a black-box result into an interpretability one.
2. **Cross-lingual transfer** — does an English-calibrated controller survive translation? MGSM gives ten languages with no data collection. Expected finding: the abstain region grows in low-resource languages, so test-time compute helps least exactly where small models are most needed.
3. **Cross-domain transfer** to execution-verified code and multi-hop QA — is the winning action a property of the problem or of the model?

None requires a rewrite: language, benchmark, policy, verifier, and now **backend** are config parameters from Day 6.
