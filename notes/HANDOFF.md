# Current state handoff

**This file is a living snapshot, NOT an append-only dated log.** Update
in place; don't accumulate history here — the dated logs
(`notes/YYYY-MM-DD.md`) already hold that. Rewritten clean 2026-08-24
after it grew append-heavy; if you're picking this up fresh, this file
alone should orient you without needing to read the dated logs first
(read them for evidence/detail, not to find out what state things are in).

Last updated: 2026-08-26 (Day 13 additions layered onto the 2026-08-24
rewrite — see the 2026-08-26 marked blocks below for what changed).
**Read this whole file before doing anything.**

## Where we actually are right now

Roadmap position: **Day 13, in progress, with Day 14's core E5/G6/H2
work pulled forward and done** (real grouped-CV macro-AUROC 0.88, H2
accepted — see "Real findings" below; user has a hard 3-4 real-day
deadline for the whole remaining project as of 2026-08-26, roadmap
compression is now the operating reality, not a hypothetical). Both dev pools are fully
generated and fully scored (a first for this project — no more waiting
on generation for anything currently planned). Gate G9 has passed.
**FREEZE is decided but not yet executed** — held over from Day 12
specifically to fold in the Day 13 temperature-ablation finding and its
band-landscape caveat (see "Real findings" and "Git state" below) before
the commit/tag happen, per explicit instruction that a check meant to
inform a frozen decision has to gate it, not follow opportunistically.
Still the single most important pending action.

### The two load-bearing decisions (frozen, only revisable by explicit new instruction)
1. **Gate G1 FAILED for the primary policy (Qwen3.5-4B).** Oracle-vs-
   best-fixed-policy accuracy gap is ~1pp against an 8pp accept bar —
   confirmed repeatedly, most recently at full 500-problem scale with a
   95% CI of `[0.20, 1.60]` (genuinely excludes zero now, same point
   estimate as the original smaller check). Logged as the project's
   pre-committed negative result. Full trail: `notes/2026-08-21.md`,
   `notes/2026-08-22.md`.
2. **Action space is a TRUE 4-class oracle label (STOP/SAMPLE/SELECT/
   ABSTAIN)**, matching docs/brief.md §16's literal spec. SELECT's real
   win rate: **0.8%** (4/500 on P1 alone; 6/754 = 0.8% on P1∪P2
   combined) — confirmed FOUR independent ways now (oracle ceiling on
   4B, weaker-policy diagnostic on 2B, real PRM reproduction with ~zero
   flips, and this 4-class relabeling). SELECT was never hard-excluded
   from the label space (that was an earlier, since-corrected draft) —
   it's a real 4th class that the data itself shows is almost never
   correct, which is a stronger, more defensible artifact than an a
   priori exclusion. Full trail: `notes/2026-08-23.md`, `notes/2026-08-24.md`.

**Neither decision has mentor confirmation.** Both were made by the user
without waiting for it, explicitly, because waiting wasn't practical.
This is disclosed, not hidden — see the mentor-brief doc below.

## What's actually built and tested (all of it, real, live-verified)

`answers/`, `backends/`, `pools/`, `generation/` (incl. `run_sweeps.py`,
supports both MATH-500 and OlympiadBench fetchers), `budget/`,
`evaluation/` (bootstrap/McNemar/Holm-Bonferroni/difficulty bands),
`selectors/` (all 4 MUST selectors), `scoring/` (segmentation + PRM
client + resumable pipeline), `telemetry/`, `controller/` (featurize,
4-class oracle labels, `DetectiveController` predictor,
`FortuneTellerController` pre-hoc control, all 7 fixed policies from the
brief's E7 design), `replay/`, `gateway/`, `answers/thread_safety.py`
(general fix for a real main-thread-only crash — see below). **110/111
tests passing** (1 intentional skip: local-backend determinism, never
exercised, honestly skipped not faked).

`search/beam.py` (bounded PRM-guided beam search, A3) **exists but is
NOT trustworthy yet** — see "What's broken" below. Do not run G5 or
anything larger on it until it's fixed.

## Real data state

- **P1 (500 MATH-500 problems, N=32, 16,000 samples): fully generated
  AND fully PRM-scored.** Zero failures on either pass (after one retry
  each for a handful of genuine timeout failures). Lives in
  `results/pools/` and `results/scores/`. No logprobs (generated before
  the backend fix below).
- **P2 (300 OlympiadBench-A problems, N=32, 9,600 samples): fully
  generated AND fully PRM-scored.** Same clean result. **Has real
  per-token logprobs** (generated after the backend fix) — see the
  memory-safety note below, this matters.
- Both pools took real hours (P1: 14.5h, P2: ~13h) — every time
  estimate this project has made has undershot actual elapsed time by
  2-4x, consistently, not a one-off pattern.

## Real bugs found and fixed this session (chronological, all with regression tests)

1. **4 real `math_verify` usage bugs** (Day 3) — silent set-construction
   on disagreeing boxed answers, truncation-fallback ordering, bare-
   LaTeX mis-parsing on both prediction and gold sides.
2. **Tie-handling inconsistency** (Day 5→8) between `selectors/basic.py`
   and the actual G1 gate script — resolved via a canonical `accuracy()`
   aggregator.
3. **Threading crash in scoring** (Day 6) — `math_verify` parallelized
   across threads crashes (`signal.alarm()` is main-thread-only).
4. **Missing logprobs request** (Day 10) — `HostedQwen35Backend` claimed
   support but never asked for them. P1 has none; P2 (generated after
   the fix) does.
5. **The SAME threading crash again, in the live gateway** (Day 10) —
   Starlette runs sync routes in a worker thread by default, so every
   real `/solve` request touching answer extraction would have crashed
   in production. Fixed generally this time via
   `src/marginal_token/answers/thread_safety.py` — **any future
   `answers/` call site that might run off the main thread must use
   these wrappers, not bare `math_verify`.**
6. **`.gitignore` path mismatch** (Day 10/freeze-prep) — excluded
   `data/pools/**`, which never matched anything; real pool/score data
   lives at `results/pools/`/`results/scores/` (17GB). Fixed before it
   could get committed. `.gitignore` now correctly excludes both
   `results/pools/**` and `results/scores/**` (figures are NOT excluded
   — those are meant to be committed).
7. **Memory-safety incident** (Day 12) — a local-analysis script
   batch-loaded all ~800 P1+P2 pools into memory at once. Harmless for
   P1 (no logprobs) but P2's real per-token logprob arrays made this use
   9.4GB+ RAM and fill all 4GB of swap on this **13GB machine**, caught
   live by the user noticing the laptop struggling. Killed immediately,
   rewritten to stream one pool at a time (~400-500MB peak instead).
   **Standing rule now: any script touching P2 (or any future
   logprob-bearing pool) must process one pool at a time, never
   batch-load across many problems.** Check `free -h` before launching
   anything that loads bulk pool data.
8. **Search arm finish-detection bug** (Day 12) — used `extract_answer`'s
   plain-text fallback (meant for salvaging truncated completions),
   which spuriously matched stray math mid-reasoning as a "finished"
   answer, stopping search after one step. Fixed by requiring an actual
   `\boxed{}` span first (`find_boxed_spans`).

## What's broken and NOT safe to run further yet

**The search arm's continuation design doesn't work.** After fixing bug
#8 above, live-tested again on 3 real problems: the model, when
re-prompted with "here's the problem + your partial solution so far,
continue," **repeats its opening sentence instead of progressing** —
confirmed on all 3 test problems, never reached a real boxed answer in
8 rounds on any of them. The PRM correctly detects this (scores decline
round over round on the worst case). This needs real design iteration
(a fundamentally different continuation strategy), not a quick patch.

**2026-08-26 DECISION: G5's documented fallback invoked, SEARCH dropped.**
Per docs/roadmap.md Day 12's own G5 fallback ("if still broken by Day
14, drop SEARCH entirely and report as a negative engineering result")
— invoked one day early (Day 13, not Day 14) since the continuation
design's failure mode is a real redesign, not a shallow bug, and the
project is already running 2-4x behind every estimate. Roadmap's own
fallback text says this makes the action space "3-way" — that phrasing
predates the 2026-08-23/24 4-class reversal and is stale; SEARCH was
already excluded from the oracle label space as of that decision, so
**the practical action space going forward stays 4-way
(STOP/SAMPLE/SELECT/ABSTAIN)**, unchanged by today's SEARCH drop —
today's decision is about the controller/live-gateway action roster
(SEARCH was never going to be offered live either way now), not a
further narrowing of the oracle labels. G5's 20-problem check and the
full 100-150-problem run are formally not happening. Script:
`notes/scratch/day12_search_smoke_test.py` (the last real evidence —
never re-run after a fix, since no fix was attempted).

## Real findings from today worth carrying forward

- **E2 landscape (P1∪P2, 754 problems, by difficulty band): the tiny
  G1 gap is NOT uniform — it's concentrated almost entirely in one
  band.** The hardest 20% of problems have a **0.000 ceiling for every
  action, including the oracle** (nothing works, no accuracy to
  reallocate). The easiest 40% are already ~1.000 for everything. One
  moderately-hard band carries ~4pp of gap on its own — a much better
  story than the flat ~1pp aggregate. Script:
  `notes/scratch/day12_e2_landscape.py` (the memory-safe version).
  **2026-08-26 CORRECTION (upgraded from caveat to demonstrated finding
  — read before repeating either claim to the mentor):** both the
  "0.000 ceiling, nothing works" (band 0) and "carries ~4pp of gap"
  (band 1) framings are largely a **token-budget artifact, not a clean
  reasoning-headroom signal**. `oracle_action_label`'s majority
  computation silently excludes failed extractions (`length_truncated`)
  from the vote; on band 0, 99% of problems average only 0.21/32
  successful extractions (the model essentially never finishes within
  1024 tokens), and on band 1, 42% of problems have ≤3/32 successful
  extractions. Bands 2-4 (60% of all problems, where the aggregate
  G1/SELECT verdicts are actually decided) are completely unaffected.
  **Demonstrated, not just inferred:** regenerated the 6 most extreme
  band-0 problems (pass@1=0.000, 0/32 extractions at 1024 tokens) at
  max_tokens=4096, temp held at the frozen 0.8. **6/6 flipped from
  ABSTAIN to a real action** (5 all the way to STOP, the cheapest
  action); mean successful extractions went from 0% to ~75%. "0.000
  ceiling, nothing works regardless of strategy" should read "...at
  max_tokens=1024" — these are not unsolvable problems, the model was
  being cut off mid-derivation every time. **The aggregate G1-failed /
  SELECT-0.8% verdicts are NOT called into question by this** (bands
  2-4 are unaffected) — but the band-0/1 *characterization* needs this
  correction, and a systematic max_tokens sweep across all of band 0/1
  is now a real open item before the report's headline numbers are
  final (see Pending decisions). Full writeup: `notes/2026-08-26.md`.
- **Predictor accuracy is misleading in aggregate** (original 2026-08-24
  finding, in-sample 100-problem preliminary). **2026-08-26 UPDATE: the
  real Day-14 evaluation this called for has now been run — DONE, not
  just called for.** Real grouped-by-problem, stratified-by-benchmark
  5-fold CV, full 787-problem scale (`notes/scratch/day13_e5_predictor_cv.py`):
  **macro-AUROC = 0.8797, clears G6's ≥0.70 accept bar comfortably. H2
  accepts.** Per-class AUROC: STOP 0.9955, ABSTAIN 0.9598, SAMPLE 0.8642
  (real signal exists), SELECT 0.6994 (n=9, structurally too rare for a
  robust number). Overall accuracy 90.1% vs. 54.6% trivial-majority
  floor — real 35.4pp lift. The imbalance concern was correct but
  narrower than it looked: SAMPLE's AUROC is genuinely good (0.86) even
  though its argmax recall is only 16.2% — a calibration issue (the hard
  decision under-uses real signal), not "no signal." Tried
  `class_weight='balanced'` as a fix
  (`notes/scratch/day13_e5_predictor_cv_balanced.py`): real trade-off,
  not a clean win — SAMPLE recall 16.2%→55.9% but ABSTAIN recall
  96.1%→65.7%, overall accuracy 90.1%→77.6%, SELECT recall stays 0%
  either way (confirms structural, not a reweighting problem). **Decision:
  report the plain model as the headline result, document balanced as an
  ablation.** This is real Day-14/E5/G6 progress, pulled forward under
  the 3-4-day time constraint. Still outstanding: H3 (needs the Fortune
  Teller pre-hoc comparator), the other 3 comparators from the brief's
  5-comparator table, probe-size ablation (SHOULD). Full writeup:
  `notes/2026-08-26.md`.
- **Temperature was never varied.** Fixed at 0.8 (top_p 0.95) for
  literally every sample generated this entire project. A small,
  targeted diagnostic (~20-25 problems, N=32, same everything except a
  higher temperature, compared paired against the same problems at 0.8)
  was proposed and agreed as a good idea — **NOT yet started.** This is
  a good candidate for the next session to actually run, before the
  freeze if you want it to inform that decision.
- **Oracle action distribution (P1∪P2 combined, 754 problems):** STOP
  56%, ABSTAIN 36%, SAMPLE 7%, SELECT 0.8%. Worth internalizing: over a
  third of all problems are unwinnable by ANY strategy (ABSTAIN is
  oracle-correct only because nothing else works either) — this is a
  big part of why the aggregate accuracy-based gap is so small, and it's
  a genuinely different claim from "smart allocation doesn't help" — see
  the ABSTAIN-vs-fixed-policy discussion in this session's chat if it's
  not in the dated notes yet.

## Pending decisions — the actual open items

1. **Execute the freeze** (`git add`/commit/`git tag design-frozen`) —
   held over from Day 12 to fold in Day 13's temperature-ablation
   caveat first (done, see above); also needs to fold in the git-state
   correction (see "Git state" above — bundle everything untracked into
   one new commit, don't rewrite the pushed `f59ae65`). This is the
   highest-priority next action once that commit/tag is actually run.
2. **Search arm: fix vs. drop — DECIDED 2026-08-26, dropped.** G5's
   documented fallback invoked (see "What's broken" above); action space
   stays 4-way (STOP/SAMPLE/SELECT/ABSTAIN), unaffected by this.
3. **N=64 pool extension** — proposed, not started. SHOULD-tier,
   explicitly conditioned on "being on schedule" (we're not). My
   recommendation: skip for now.
4. **Temperature ablation** — DONE 2026-08-26, run before the freeze per
   explicit instruction (correcting an earlier plan to run it
   opportunistically after), in two passes. First pass (this session's
   own band-1 difficulty-band proxy): inconclusive by design — band 1
   turned out to be the one band with a real extraction-survivorship
   confound (see the caveat above). **Second pass (the mentor's actual
   suggestion, relayed mid-session): tested on the 4 real, canonical P1
   problems whose true oracle label is SELECT at temp=0.8, not a
   difficulty-band proxy.** The one clean case (25/32 successful
   extractions) is completely stable across temperature — SELECT wins
   identically at 0.8 and 1.0, real signal, not moved either direction.
   The other 3 show the same survivorship noise. **Clean answer for the
   mentor: temperature doesn't rescue SELECT and doesn't fix truncation
   — `max_tokens` is the real lever, confirmed two independent ways.**
   **Third pass: does max_tokens itself rescue SELECT?** Tested the 2
   known truncation-confounded SELECT-oracle problems at max_tokens=4096
   — both flipped to STOP with perfect 32/32 extraction, SELECT's
   apparent win vanished rather than strengthening. Combined with the
   band-0 max_tokens result (6/6 ABSTAIN->mostly STOP, none to SELECT):
   **8/8 consistent evidence that more tokens converts broken problems
   into easy ones, never into genuinely SELECT-favorable ones — a 5th
   independent confirmation SELECT is genuinely rare, not an artifact of
   temperature or truncation.** Full writeup: `notes/2026-08-26.md`.
5. **Mentor conversation** — still not confirmed to have happened. Full
   context + specific questions prepared in
   `notes/mentor-brief-summary-2026-08-24.md`.
6. **NEW 2026-08-26: systematic max_tokens sweep across band 0 (and
   maybe band 1)** — not started. The 6-problem hand-picked check above
   found 6/6 flip from ABSTAIN to a real action (5/6 to STOP) at
   max_tokens=4096 vs. 1024, ~75% mean extraction recovery from a 0%
   floor. That's strong signal on n=6 but not a claim that all of band 0
   recovers this cleanly. Real cost: these are the longest completions
   this project has generated for MATH-500 (43.5min for 6 problems x
   N=32 at 4096 tokens) — a full band-0 sweep (~100-151 problems) would
   be a genuinely large time commitment, on a project already 2-4x over
   every estimate. My recommendation: worth doing before the report's
   headline band-landscape numbers are finalized (it changes a claim
   currently stated as fact — "0.000 ceiling, nothing works"), but NOT
   worth doing before the freeze/mentor conversation — raise it as a
   real open item for the mentor to weigh in on scope/priority, same as
   the search-arm and N=64 questions, rather than unilaterally launching
   a multi-hour run.

## Infrastructure facts (don't rediscover these)

- **This machine has only 13GB RAM, 4GB swap.** Real constraint, hit
  once already (see bug #7). Check `free -h` before anything that loads
  bulk data.
- **Working directory is nested**: the actual git repo is
  `~/Desktop/marginal-token-skeleton/marginal-token/` — one level deeper
  than the outer folder name suggests. `cd` there before any git command.
- **Hosted endpoints**: `policy_primary` (Qwen3.5-4B, via a third-party
  unsloth GGUF conversion, NOT official weights — flagged to mentor via
  the brief summary), `prm_primary` (Qwen2.5-Math-PRM-7B, real, fast,
  zero failures across thousands of calls), `policy_secondary_stretch`
  (2B, used once diagnostically). Full roster:
  `configs/backends/hosted-endpoints.yaml`. API key: `.env` →
  `HOSTED_ENDPOINT_API_KEY` (gitignored, confirmed excluded).
- **Raw GPU SSH**: `root@194.149.138.21` works, but its one GPU is at
  98% VRAM running the standing hosted endpoints themselves — not spare
  capacity. `.12` has no working credential.

## Git state

**2026-08-26 correction: the claim below ("nothing committed from Day 4
onward") is stale/wrong as of this update — flagging the correction
inline rather than silently rewriting history.** A commit (`f59ae65`,
2026-08-25 03:05:31, already pushed to `origin/main`) exists whose
message describes Day 10-12 conclusions but whose actual diff is
roughly Day 6-9-era infra content — origin of the mismatch unknown, not
created by any session with access to this file. `CLAUDE.md`,
`Makefile`, all Day 10-13 dated notes (including this file and the
mentor-brief), `search/beam.py`, its tests, and the E0 figures are all
still untracked as of this update. Decision: leave the pushed commit
as-is (don't rewrite pushed history over a message/content mismatch
with no functional harm — the diff content itself is real and correct),
bundle everything currently untracked into one new, accurately-described
commit instead, then apply `design-frozen` to that. `.gitignore` is
fixed (bug #6 above) — verify with `git add -n results/` that only
`results/figures/` would be staged, never `results/pools/`/
`results/scores/`, before committing.

## Read order for a fresh session

1. `CLAUDE.md`, `docs/brief.md`, `docs/roadmap.md`.
2. This file, in full — it should be sufficient on its own for current
   state.
3. `notes/mentor-brief-summary-2026-08-24.md` if prepping for the mentor
   conversation specifically.
4. Dated logs (`notes/2026-08-21.md` through `notes/2026-08-24.md`) only
   if you need the evidence trail behind a specific claim above, not to
   figure out what state things are in.
