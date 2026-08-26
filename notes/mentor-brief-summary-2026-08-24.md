# Mentor brief supplement: what actually happened, Days 1-12

Standalone summary, not part of the append-only daily logs. Purpose:
give the mentor the real story (decisions made under real constraints,
real findings) rather than presenting the original frozen brief as if
none of this happened. Load-bearing points only — full detail lives in
`notes/2026-08-18.md` through `notes/2026-08-24.md`.

## Day 1 — Setup, Gate G0
- Confirmed `Qwen/Qwen3.5-4B` as the correct HF slug (post-trained chat
  repo). Local vLLM confirmed as a working backend.
- Surveyed 6 hosted-API providers: none served this exact model at
  launch (too new). **Waived the "≥2 backends" requirement** for
  Qwen3.5-4B specifically, as an accepted risk with a compensating
  control (rent a cloud GPU rather than substitute the policy model if
  local access is lost). No model change.

## Day 2 — Provenance discipline
- Froze the held-out sets (OlympiadBench slice B, 100 problems; AIME25,
  30 problems), SHA-256 hashed, and verified the split is reproducible
  from scratch (independently re-fetched and re-derived, byte-identical
  hash match) — not just self-consistent with what was written.
- Pinned exact commit SHAs for the policy model, both PRM rungs, and
  both benchmark datasets — pool-identity discipline established before
  any real generation.

## Day 3 — Highest-risk day: real infrastructure, real bugs, a frozen-invariant change
- **Mentor provided real hosted infrastructure mid-day** (policy 4B + 2B,
  both PRM rungs, raw GPU servers). G0 formally un-waived.
- **Found and flagged immediately:** the hosted policy endpoint serves a
  third-party `unsloth` GGUF conversion via llama.cpp, **not** the
  official HF safetensors via vLLM. Documented as a pool-identity-
  relevant condition, not silently treated as equivalent. (You said
  you'd raise this with the mentor separately — flagging again here only
  because it belongs in the full story.)
- Golden-200 hand-check (200 real completions, hand-verified) surfaced
  and fixed **4 real correctness bugs**, all in how `math_verify` was
  being used — not hypothetical edge cases: silent set-construction on
  disagreeing `\boxed{}` values, a truncation-fallback ordering bug that
  could credit a stray mid-derivation equation as the final answer, and
  bare-LaTeX mis-parsing on both the prediction and gold sides
  (`3\sqrt{5}` silently becoming `3`, a tuple silently collapsing to one
  element). All four fixed with regression tests before trusting any
  downstream number.
- **Truncation finding:** at the frozen `max_tokens=1024`, 32% of
  MATH-500 and 74% of OlympiadBench-A completions never reach a boxed
  answer. Re-tested at 4x budget (4096): OlympiadBench-A still 50%
  truncated — a heavily diminishing-returns curve, no `max_tokens` value
  the data supports recommending. Flagged, not silently fixed by
  bumping a frozen decode config.
- **Gate G2** (backend generates end-to-end within a 4h target):
  qualitative pass, quantitative miss — real measured throughput implies
  ~5.6h for Day 4's actual MATH-500×N=64 workload. You accepted the
  deviation; no backend change, no scope cut.
- **N=64 → N=32 MUST floor changed by your explicit instruction** —
  permanent, project-wide (CLAUDE.md, brief.md, roadmap.md all updated,
  full workload-ledger recomputation, ~27 touch points). Budget levels
  became `{2,4,8,16,32}`; the old N=128 SHOULD-extension became N=64.

## Day 4 — Baseline reproduction, Gate G1 (the load-bearing gate)
- Generated real pools: MATH-500 (100 problems × N=32, extended to
  N=64) and OlympiadBench-A (30 problems × N=32, 908/960 samples after
  a real operational incident — orphaned in-flight requests from a
  killed run competing with the retry; 2 problems dropped entirely on
  persistent server-side 500 errors).
- **Gate G1 (oracle-over-actions must beat the best fixed policy by
  ≥8pp): FAILED, on every check run.** MATH-500 N=32 gap = 1pp, N=64 gap
  = 2pp, OlympiadBench-A gap = 0pp exactly. Ran the full documented
  fallback chain, not just the primary check: a weaker policy
  (Qwen3.5-2B) diagnostic showed a real but modest 4pp gap at full scale
  (n=100) — a noisy cheap check first suggested 12pp, which the full-scale
  run showed was optimistic noise, not signal. Still short of 8pp.
- **Mechanism found, not just the null result:** the model's running
  majority vote locks onto its final answer almost immediately for most
  problems (mean stabilization point k*=2.17 samples out of 32). This
  explains *why* G1 is null (STOP and SAMPLE mostly agree, leaving an
  oracle little room to gain) and is itself a separate, real
  **cost-efficiency** finding — large headroom to stop early without
  losing accuracy — a different question from G1's accuracy-headroom
  question, not a contradiction of it.
- **Two decisions made without waiting for your reply**, both logged as
  revisable pending this conversation:
  1. **G1 verdict: FAILED for the primary policy** — logged as the
     project's pre-committed negative result (per the brief's own §34).
  2. **Action space narrowed from {STOP, SAMPLE, SELECT, ABSTAIN} to
     {STOP, SAMPLE, ABSTAIN}.** SELECT's oracle win rate measured at only
     1% (4B) / 3% (2B, full scale) — both under the 5% threshold set in
     advance, on two independent checks. Same precedent the brief
     already uses for SEARCH ("if it never wins, narrow, report as a
     negative result"). PRM scoring itself was **not** dropped — it
     remains a planned predictor feature independent of SELECT's status
     as a controller action.

## Day 5 — PRM integration, Gate G3, B3 reproduction
- Found the brief's own assumption ("no hosted API provides PRMs") was
  already stale — the mentor's endpoint roster included a real hosted
  PRM scorer. No GPU needed for PRM scoring after all.
- **Gate G3 (PRM step scores must predict correctness, AUROC>0.6):
  PASSED decisively** (0.9934), using `double_newline` step segmentation
  — the only one of the three documented conventions with real coverage
  for this policy's actual output style (it uses numbered markdown, not
  literal "Step k:").
- **B3 reproduced at full scale (100 problems): PRM-weighted majority =
  plain majority exactly (0.730 both), zero individual-problem flips.**
  A real PRM — not just the oracle-proxy ceiling from Day 4 — also finds
  essentially nothing to do differently from plain majority on this
  data. This is an independent, second confirmation of the Day-4
  SELECT-narrowing finding, from a completely different measurement.

## Day 6-9 — Real infrastructure build, two real production bugs, P1 generated
- Built the actual codebase for real: backend abstraction, resumable
  generation, content-addressed pool store, offline PRM scoring, all 4
  MUST selectors, statistics (bootstrap/McNemar/Holm-Bonferroni), all
  live-tested against real data, not just unit tests.
- **P1 (500 MATH-500 problems, N=32, 16,000 samples) fully generated** —
  14.5 hours real wall-clock (vs. the brief's a-priori 4-7h *combined*
  P1+P2 estimate — every real time estimate this project has made has
  undershot actual elapsed time by 2-4x, consistently, not a one-off).
- Two real, previously-undiscovered bugs, both caught before they
  propagated: (1) a threading bug where `math_verify`'s timeout crashed
  when parallelized — found once in a scoring script, fixed there, then
  found **again** in the actual live gateway's `/solve` route (Starlette
  runs sync routes in a worker thread by default — every real production
  request would have crashed). Fixed generally this time. (2) The hosted
  backend claimed to support per-token logprobs but never actually
  requested them — P1's entire pool has none; fixed for future
  generation, not retroactive.

## Day 10 — Controller built, SELECT question revisited and strengthened
- Built the real controller: probe features, oracle action labels, a
  logistic-regression predictor ("Detective"), a pre-hoc query-embedding
  control ("Fortune Teller," non-negotiable per the brief for H3's
  falsifiability), and all 7 fixed comparator policies from the brief's
  own E7 design.
- **Revisited the Day-4 SELECT-narrowing call, made it stronger, not
  just repeated it.** Instead of hard-excluding SELECT from the oracle
  label space (a real deviation from the brief's own literal "primary
  4-class" spec), rebuilt it as a TRUE 4-class label (STOP/SAMPLE/SELECT/
  ABSTAIN) and let the data show SELECT's rate directly: **0.8% (4/500)
  on the full P1 pool** — matching the earlier 1%/3% estimates almost
  exactly, but now as a measured result of the actual spec-compliant
  design, not an a priori exclusion.
- **P2 (300 OlympiadBench-A problems, N=32, 9,600 samples) also fully
  generated and fully PRM-scored** — another ~13 hours real wall-clock.
  Both dev pools (P1 + P2) are now complete end to end.

## Day 11 — E0 thin slice, Gate G9
- Ran the full Diagnose→Predict→Allocate chain on the 100-problem dev
  slice. **Gate G9 passes** (its bar is just "did E0 produce H1/H2/H4
  numbers" — it did, with real figures).
- H4's preliminary result: the learned controller beats neither fixed
  policy at any of 5 budget levels (0/5, needs ≥3/5 to accept H4). Not a
  new surprise — flagged as the anticipated outcome back on Day 5,
  since H4 is mathematically bounded by H1's own small gap.

## Day 12 — E2 landscape: the gap has real structure, just not where it's big
- Full P1∪P2 landscape (754 usable problems) by difficulty band. **Real
  finding: the tiny aggregate G1 gap is not uniform — it's concentrated
  almost entirely in one difficulty band.** The hardest 20% of problems
  have a **0.000 ceiling for every action, including the oracle** —
  nothing works there regardless of strategy, so there's no accuracy to
  reallocate toward on those. The easiest 40% are already at ~1.000 for
  everything. The one moderately-hard band in between carries ~4pp of
  gap on its own — a much more informative story than "the gap is ~1pp."
- **Started building the search arm (A3/SEARCH, bounded PRM-guided beam
  search) and found it's not ready.** Two real bugs caught live, in the
  first two smoke tests: a finished-detection bug that stopped search
  after one step (fixed), and — after fixing that — a real, unfixed
  problem where the continuation design makes the model repeat its
  opening sentence instead of progressing, across all three test
  problems, never reaching a real answer. **Not launching G5's real
  20-problem check on this yet** — it needs real design iteration, not
  a rushed unsupervised run.
- Also hit a real operational scare: an early version of the E2 script
  loaded too much pool data into memory at once (P2's samples carry real
  per-token logprob data now) and used 9.4GB+ RAM, filling all swap on
  this 13GB machine. Caught live, killed immediately, rewritten to
  process one problem at a time — no data lost, but worth mentioning
  since it's a real resource constraint on this project's actual
  hardware, not just a cloud-compute abstraction.

## Day 13 — Temperature ablation, and a bigger finding than the one it was looking for
- Ran the temperature ablation agreed on Day 12 (21 MATH-500 problems,
  N=32, temp=1.0 vs. the frozen 0.8), deliberately targeting the one
  difficulty band (band 1) where Day 12 found real gap to test —
  otherwise there'd be nothing to detect either way.
- **Raw result looked dramatic** (oracle-vs-fixed gap 14.29pp at
  temp=0.8 vs. 4.76pp at temp=1.0, paired CI excluding zero) but turned
  out not to be a real temperature effect. Investigation traced it to a
  code behavior: the oracle-label majority computation silently drops
  failed extractions (`length_truncated`) from the vote. On
  truncation-heavy problems this makes "the majority" trivial — computed
  among whichever 1-3 samples (out of 32) happened to produce a boxed
  answer at all, not a genuine majority-of-32.
- **Checked how widespread this is across all of P1, by band:** band 0
  (hardest 20%) averages only 0.21/32 successful extractions (99% of
  problems affected); band 1 averages 7.06/32 (42% affected); bands 2-4
  (60% of all problems) are completely clean (0% affected). This means:
  - Day 12's "band 0 has a 0.000 ceiling, nothing works there regardless
    of strategy" is likely mischaracterized — it reads much more like
    "the 1024-token budget is too short to finish these problems" than
    a genuine reasoning-difficulty ceiling.
  - Day 12's "band 1 carries ~4pp of gap" is likely partly a
    survivorship artifact from the same mechanism, at smaller scale.
  - **The aggregate G1-failed / SELECT-0.8% verdicts are NOT called into
    question** — both are decided by bands 2-4, which are unaffected.
    Only the band-landscape *mechanism story* needs a correction.
- Salvaged a real, clean answer from the same data even though the
  original ablation question came back inconclusive: temperature barely
  moves the truncation rate itself (2.67/32 -> 2.81/32 successful
  extractions, 0.8 vs. 1.0) — `max_tokens`, not temperature, is the
  actual lever on band-0/1 behavior, if that's ever revisited.
- Did not attempt a "clean" rerun on a low-truncation band (e.g. band 2)
  as a workaround — checked first that band 2's own gap is already only
  0.66pp with clean extraction, so a rerun there would trivially show
  "no effect" because there's nothing to detect, not because temperature
  doesn't matter. Full writeup: `notes/2026-08-26.md`.
- **Separately, found a real discrepancy in git state** while checking
  what freezing would touch: a commit (`f59ae65`, already pushed to
  `origin/main`) has a message describing Day 10-12 conclusions but a
  diff containing only Day 6-9-era content — `CLAUDE.md`, this file, all
  Day 10-13 notes, and the search-arm code/tests are all still
  untracked. Origin unknown. Decided not to rewrite the pushed commit;
  bundling everything untracked into one new, accurately-described
  commit instead, then applying `design-frozen` to that.
- **Ran the mentor's own suggestion directly, not just this session's
  difficulty-band proxy:** on hearing about SELECT's near-zero win rate,
  the mentor suggested raising temperature, using a smaller pool, and
  testing specific problems expected to favor SELECT. Redone properly on
  the 4 real, canonical P1 problems whose true oracle label is SELECT at
  temp=0.8 (not a difficulty-band proxy). **Answer: temperature doesn't
  rescue SELECT.** The one clean, uncontaminated case (25/32 successful
  extractions) shows SELECT winning identically at both temp=0.8 and
  temp=1.0 — real, stable signal, unmoved by temperature in either
  direction. The other 3 (all extraction-confounded) just relabel
  noisily based on which 2-5 of 32 samples happened to complete, and one
  hints (n=1, weak) that higher temperature may make truncation
  slightly worse, not better. `max_tokens`, not temperature, is the
  real lever here — confirmed two independent ways today.
- **Followed up: does max_tokens itself rescue SELECT?** Tested the 2
  known truncation-confounded SELECT-oracle problems at max_tokens=4096.
  **Both flipped to STOP with perfect 32/32 extraction — SELECT's
  apparent win vanished rather than strengthening.** Combined with the
  band-0 max_tokens result, this is 8/8 consistent evidence that more
  tokens converts broken problems into easy ones, never into genuinely
  SELECT-favorable ones — a 5th independent confirmation that SELECT is
  genuinely rare in this data, not an artifact of temperature or
  truncation. Confident answer to give the mentor on this specific
  point.

## Day 13 continued — the max_tokens finding (bigger than the ablation that found it)
Prompted by a plain-language worry mid-session: most problems resolve to
STOP or ABSTAIN — is the controller actually doing anything useful?
Traced to the same root cause as the extraction-survivorship finding
above, then tested directly instead of just theorizing:

- Picked the 6 most extreme MATH-500 problems available: pass@1=0.000
  AND 0/32 successful extractions at the frozen max_tokens=1024 —
  completely dead, zero information, at the current budget.
- Regenerated all 6 at max_tokens=4096 (temperature held at the frozen
  0.8, isolating max_tokens as the only changed variable).
- **6/6 flipped from ABSTAIN to a real action.** Mean successful
  extractions went from 0% to ~75%. 5 of 6 resolved all the way to STOP
  (the cheapest possible action); the 6th to SAMPLE. One problem went
  from 0/32 to a perfect 32/32.
- **This upgrades Day 12's "hardest 20% has a 0.000 ceiling, nothing
  works there regardless of strategy" from a caveated claim to a
  demonstrated correction: it should read "...at max_tokens=1024."**
  These are not unsolvable problems — the model was being cut off
  mid-derivation on every sample, every time. Doesn't contradict Day 3's
  aggregate truncation finding (OlympiadBench-A still 50% truncated at
  4x budget) — that was an aggregate average across all difficulty
  levels; this targeted the single most extreme dead tail of MATH-500
  specifically. Both can be true at once.
- **Direct answer to the controller-usefulness worry:** a real chunk of
  the 36% ABSTAIN mass in the oracle action distribution isn't "no
  headroom exists" — it's "the harness never let the model finish." The
  controller's actual decision-relevant workload at a properly-sized
  budget is likely very different from what's measured at 1024 tokens.
  Doesn't mean the controller implementation itself is fine (Detective's
  0% recall on true SELECT cases is still real and separate) — but part
  of what looked like "nothing to do" was a budget artifact.
- n=6, hand-picked (deliberately the worst case) — not a claim the
  whole of band 0 recovers this cleanly. A systematic max_tokens sweep
  across band 0 (and maybe band 1) is a real open item, explicitly not
  yet started and not run before this freeze — see the question below.

## Where things stand right now
- **FREEZE (`design-frozen` tag) has been decided** — held one more day
  (Day 13) specifically to fold in the temperature-ablation finding and
  its band-landscape caveat above before committing/tagging, per
  explicit instruction that a check meant to inform a frozen decision
  has to gate it, not follow opportunistically. This locks in the
  4-class action space and everything above, now including the Day 13
  correction.
- P1 and P2 are both fully generated and fully scored. G9 has passed.
  The search arm is the one piece of Week 2/3's MUST-tier work that's
  genuinely not ready yet (dropped, see Day 12 above and the roadmap's
  own G5 fallback).

## The throughline
Every real check across two weeks — the oracle-ceiling measurement, the
weaker-policy diagnostic, the real PRM reproduction, and now the full
4-class relabeling at 5x the original data with a tighter CI — converges
on the same conclusion: for this policy, on this data, answer-selection
methods (SELECT) have essentially no headroom over plain majority
voting. **Where that headroom is concentrated is less settled than Day
12 made it sound** — Day 13 found the band-1 "concentration" finding is
likely partly a token-truncation artifact, not clean reasoning-headroom
signal (see above); the honest statement is "no headroom in aggregate,
and what little exists doesn't survive a clean look at the one band
that seemed to carry it." What *does* have real, measured,
well-replicated headroom is **when to stop sampling** (the k*≈2 finding,
confirmed again at full scale) — a cost-efficiency question, not an
accuracy one, and the project's actual positive result.

## Questions worth actually asking the mentor
1. **Sign-off on the SELECT-narrowing decision**, now on its strongest
   evidence (4-class relabeling, 0.8% win rate, 4 independent
   confirmations, a G1 CI that now excludes zero). This was decided
   without waiting for input, twice (Day 4 and again Day 10) — worth
   an explicit "does this match your read" before it's permanently
   frozen, not after.
2. **The GGUF/llama.cpp-via-unsloth vs. official HF-safetensors
   caveat** — flagged internally since Day 3, never actually raised
   with the mentor. Every real number in this project comes from the
   third-party conversion, not the official checkpoint. Worth raising
   now, plainly, not deferred further.
3. **Is freezing without a mentor check-in on this decision the right
   call for how this course/project expects sign-off to work?** The
   science is solid; whether the *process* is fine is not something the
   data can answer.
4. **The search arm (A3) is behind schedule and has a real, unresolved
   design bug (model repeats itself instead of continuing a partial
   solution).** Given how far every other estimate has already overrun,
   is it worth the real additional time to iterate on this properly, or
   should the documented G5 fallback ("if broken by Day 14, drop A3,
   report as a negative engineering result, 3-way action space") be
   invoked earlier rather than later?
5. **Given the consistent 2-4x overrun on every time estimate so far**,
   is the current scope (full P1 500 + P2 300, N=32, plus the search
   arm, plus held-out P4+P5) still realistic, or does the mentor want to
   invoke the brief's own documented scope-reduction path (§27.5:
   OlympiadBench 300→200, search arm held at the 100-problem floor) now
   rather than discovering the crunch later?
6. **Is N=64 pool extension worth doing at all**, given it's SHOULD-tier
   and explicitly conditioned on "being on schedule" (which this project
   isn't)? A real ~13-14h-per-pool commitment either way.
7. **Does the Day 13 extraction-survivorship correction change how the
   band-landscape result should be reported**, given the headline
   G1/SELECT verdicts survive it untouched but the "gap concentrated in
   one band" / "hardest band is truly unwinnable" framing doesn't? Is a
   caveated version of the band story still worth including in the
   final report, or is it cleaner to drop the band-landscape framing
   entirely and report only the clean aggregate result plus the k*≈2
   cost-efficiency finding?
8. **Is a systematic max_tokens sweep across band 0 (and maybe band 1)
   worth the real time cost before the report is final?** The 6-problem
   hand-picked check found 6/6 flip from ABSTAIN to a real action (5/6
   all the way to STOP) at 4096 vs. 1024 tokens — strong signal that
   "the hardest 20% is unwinnable" is largely a budget artifact, not a
   reasoning ceiling. But these were also the longest completions this
   project has generated (43.5min for just 6 problems x N=32), and a
   full band-0 sweep (~100-151 problems) would be a genuinely large
   commitment on a project already 2-4x over every time estimate.
   Deliberately not launched unilaterally — this feels like exactly the
   kind of scope call that should have your input given the project's
   schedule state, not something to just go run.
