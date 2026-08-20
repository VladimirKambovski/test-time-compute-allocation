# Roadmap — 20 working days

Tags: **[MUST]** required for project success · **[SHOULD]** do only after
gate G9 passes · **[STRETCH]** optional, cut first under time pressure.

Work top to bottom. Do not skip a "Done when" check. Gate outcomes get
logged in `notes/YYYY-MM-DD.md` regardless of pass/fail.

---

## Week 1 — Literature, verification, baseline, brief

### Day 1 — Novelty check + Gate G0 [MUST]
- [ ] Resolve the five flagged papers in full text: 2604.17433,
      2606.09078, 2606.08098, 2607.08065, 2506.12721 (per docs/brief.md
      §32 Day 1 plan). Write a one-page-per-paper "what remains open"
      note.
- [ ] **G0:** Verify `Qwen/Qwen3.5-4B` (exact slug!) is servable on ≥2
      backends: local via vLLM, plus at least one hosted API. Record
      pricing, whether logprobs are exposed, whether seeds are honoured.
- [ ] Configure AWS budget alarms ($10 / $25 / $40).
- **Done when:** two viable backends are named with costs, and a written
  "what remains open" line exists for each of the five papers.

### Day 2 — Literature review, held-out freeze, env [MUST]
- [ ] Draft `report/01-literature.md` (motivation, RQ, prior work).
- [ ] Select held-out sets (OlympiadBench slice B = 100 problems,
      disjoint from slice A; AIME25 = 30 problems). SHA-256 hash the
      problem ID lists. Commit the hash into
      `configs/benchmarks/heldout-*.yaml`.
- [ ] `uv sync`; pin model revisions by HF commit SHA / API model ID.
- **Done when:** the held-out hash is committed to git.

### Day 3 — Answer extraction and equivalence [MUST] — highest-risk day
- [ ] Build `src/marginal_token/answers/`: extraction, canonicalization,
      equivalence via `math_verify`.
- [ ] Hand-check 200 (prediction, gold) pairs across MATH-500 +
      OlympiadBench. Save as `tests/fixtures/golden_200.json`.
- [ ] `tests/test_answer_equivalence.py` passes on all 200.
- [ ] Record extraction-failure / ambiguity rate in `notes/`.
- [ ] **G2:** confirm chosen backend generates end-to-end within 4 hours.
- **Done when:** golden test passes; rates are logged.

### Day 4 — Baseline reproduction + Gate G1 [MUST] — most decisive day
- [ ] Generate 100 MATH-500 problems, N=32, on the primary policy.
- [ ] Compute maj@k, pass@k.
- [ ] **G1:** oracle-over-{STOP,SAMPLE,SELECT} minus best fixed policy.
      Accept if ≥8pp. See docs/brief.md §33 for the fallback chain if
      it fails (promote OlympiadBench → weaker policy → mentor escalation).
- [ ] Compare maj@k curve against published values.
- **Done when:** G1 decision is written down and the primary benchmark
  is fixed for the rest of the project.

### Day 5 — PRM integration, Gate G3, brief presented [MUST]
- [ ] Load primary PRM. Implement 3 step-segmentation conventions.
- [ ] Score the Day-4 pool.
- [ ] **G3:** PRM step scores must predict correctness (AUROC > 0.6) on
      10 problems before any scale scoring. Fallback: other
      segmentation → PRM ladder rung 2 → if exhausted, **G10** (ask,
      do not substitute a different verifier class).
- [ ] Reproduce B3 (PRM-weighted majority vs plain majority).
- [ ] Present the brief.
- **Done when:** G3 passes (or its fallback is invoked and logged) and
  the brief has been presented.

---

## Week 2 — Core implementation, early integration, freeze

### Day 6 — Backend abstraction, generation, pool store [MUST]
- [ ] `src/marginal_token/backends/`: driver interface, capability
      reporting, provenance metadata block on every sample.
      `tests/test_backend_metadata.py`.
- [ ] `src/marginal_token/generation/`: config-hashed, resumable sweeps.
- [ ] `src/marginal_token/pools/`: content-addressed store, nested-prefix
      views.
- [ ] Start P1 (MATH-500, N=32) generation in the background.
- **Done when:** a killed generation run resumes cleanly to a complete
  pool, and every sample carries full provenance.

### Day 7 — Scoring engine + budget manager [MUST]
- [ ] `src/marginal_token/scoring/`: offline batch PRM scoring, separate
      process from generation, per-step score arrays.
- [ ] `src/marginal_token/budget/`: exact accounting — policy tokens,
      PRM forwards, discarded beam tokens. `tests/test_budget_accounting.py`.
- **Done when:** the accounting test passes, including a synthetic
  discarded-beam case.

### Day 8 — Selectors, oracle, statistics [MUST]
- [ ] `src/marginal_token/selectors/`: plain majority, oracle pass@k,
      PRM-weighted majority, PRM-argmax [MUST]; self-certainty,
      length-normalized, cluster-then-vote, PRM reductions [SHOULD].
- [ ] `src/marginal_token/evaluation/`: paired bootstrap (10k, BCa),
      McNemar, Holm–Bonferroni, difficulty banding.
- **Done when:** all MUST selectors run over a real pool; bootstrap
  reproduces a known CI on synthetic data.

### Day 9 — Determinism, telemetry, smoke test, Gate G4 [MUST]
- [ ] `tests/test_determinism.py` → **G4** (local backend: byte-identity;
      API backend: record `determinism: not guaranteed`, rely on the
      frozen artifact hash instead — see docs/brief.md §27.2).
- [ ] `src/marginal_token/telemetry/`: structured decision records via
      Langfuse, trace IDs.
- [ ] Full smoke test on 50 problems: generate → score → select →
      metrics → one figure.
- **Done when:** G4 outcome is recorded either way; one figure exists
  end to end.

### Day 10 — Controller, replay, gateway skeleton, FREEZE [MUST]
- [ ] `src/marginal_token/controller/`: `featurize()`, oracle action
      labels, logistic predictor scaffold, 7 fixed policies.
- [ ] `src/marginal_token/replay/`: replay any controller over cached
      pools.
- [ ] `src/marginal_token/gateway/`: minimal FastAPI `/solve` skeleton
      calling the **same** Controller object — no UI yet.
- [ ] `tests/test_controller_parity.py` passing.
- [ ] **FREEZE:** commit + tag `design-frozen` with hypotheses, accept
      conditions, experiment matrix, statistical plan.
- **Done when:** replay and gateway demonstrably share one Controller
  instance, parity test is green, and the tag exists. No hypothesis or
  accept-condition changes after this point.

---

## Week 3 — Experiments, ablations, failure analysis

### Day 11 — E0: thin end-to-end slice + Gate G9 [MUST] — unblocks SHOULD
- [ ] Score the 100-problem slice; compute oracle action labels.
- [ ] Run the full Diagnose → Predict → Allocate chain on this slice.
      Produce preliminary H1 / H2 / H4 numbers.
- [ ] **G9:** this must pass before any expensive SHOULD work starts.
- **Done when:** one figure exists for each of H1, H2, H4, even at
  slice scale.

### Day 12 — DIAGNOSE: landscape + search arm, Gate G5 [MUST]
- [ ] E2: action-value landscape across {STOP,SAMPLE,SELECT} × 5 budgets
      × 5 difficulty bands, on P1∪P2 at N=32.
- [ ] Adapt bounded beam search (100–150 stratified problems × 2
      matched-token budgets [MUST]; 3rd budget / 200 problems [SHOULD]).
- [ ] **G5:** sanity check on 20 problems — within 10pp of SAMPLE at
      comparable tokens. Fallback: hold at the 100-problem floor; if
      still broken by Day 14, drop SEARCH entirely and report as a
      negative engineering result (the action space becomes 3-way).
- **Done when:** G5 decision is logged.

### Day 13 — DIAGNOSE: supporting analyses [MUST/SHOULD]
- [ ] E4: PRM crossover N* with CI; length/step-count regression on
      argmax winners (verifies V4).
- [ ] Inspect 30 traces where PRM-argmax picked wrong and majority was right.
- [ ] Complete the **H1 final verdict**, including whether SEARCH
      occupies a distinct region of the action space.
- **Done when:** H1 has a written accept/reject verdict with a CI.

### Day 14 — PREDICT: predictor + comparators, Gate G6 [MUST]
- [ ] E5: fit the 4-class logistic predictor; grouped 5-fold CV;
      feature ablation [MUST]; probe-size ablation [SHOULD].
- [ ] All 5 comparators, **including the pre-hoc query-text embedding
      classifier** — this is non-negotiable, it's what makes H3 falsifiable.
- [ ] **G6:** macro-AUROC ≥0.70 → accept H2; 0.60–0.70 → weak; <0.60 →
      reject and promote the negative result.
- **Done when:** H2 and H3 have written verdicts.

### Day 15 — ALLOCATE: Pareto frontier, Gate G7 [MUST]
- [ ] E7: 7 policies × 5 matched-token budgets over cached pools.
      Evaluate H4.
- [ ] E8: cost-per-correct-answer (tokens, GPU-hours, USD). Frontier
      anchor via Bedrock; local/API 32B comparator [SHOULD].
- [ ] **G7:** resource check — if >70% of budget consumed, cut the
      remaining SHOULD tier now.
- **Done when:** all four headline hypotheses (H1, H2, H3, H4) have
  written verdicts.

---

## Week 4 — Held-out, delivery

### Day 16 — Gateway final integration [MUST]
- [ ] Complete the `/solve` contract: three outcomes (answered /
      escalated / declined), anytime budget exhaustion, live-mode path.
- [ ] Re-verify `test_controller_parity.py`.
- **Done when:** a `declined` response is produced end-to-end with a
  machine-readable reason.

### Day 17 — Demo build, Gate G8 (hard freeze) [MUST/SHOULD]
- [ ] `ui/`: probe / evidence / controller / spend / outcome panels.
      Comparison mode [SHOULD].
- [ ] Benchmark mode reads cached artifacts only — zero GPU, zero API calls.
- [ ] **G8: no new experiments after today. Full stop.**
- **Done when:** someone can pick a problem and watch the full decision
  path to an outcome.

### Day 18 — Held-out evaluation, single pass [MUST]
- [ ] Generate P4 (Olympiad-B) + P5 (AIME25) at N=32.
- [ ] Run the **frozen** controller and selectors. One pass. No tuning.
- [ ] Frontier anchor calls on held-out.
- **Done when:** held-out numbers are recorded, whatever they say. No
  re-tuning permitted, full stop.

### Day 19 — Report finalization [MUST]
- [ ] Finalize `report/`: held-out results, discussion, limitations,
      What Went Well / What Went Wrong, Next Steps, Start-vs-End
      comparison, planned-vs-actual timeline table.
- [ ] Complete the literature-claim verification table (V1–V5).
- **Done when:** every hypothesis and every V-claim has a verdict.

### Day 20 — Repro, rehearsal [MUST]
- [ ] README written for a stranger with no GPU, no API key.
- [ ] `make reproduce-headline` verified from a **clean clone**. This is
      a hard completion criterion — do not skip it.
- [ ] `results/manifest.json` complete.
- [ ] Demo rehearsed twice, start to finish.
- **Done when:** clean clone reproduces headline tables; demo has run
  successfully twice in a row.
