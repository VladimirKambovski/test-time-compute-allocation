# Current state handoff

**This file is a living snapshot, NOT an append-only dated log.** Update
in place; don't accumulate history here — the dated logs
(`notes/YYYY-MM-DD.md`) already hold that. **Rewritten clean 2026-09-01**
(previous clean rewrite was 2026-08-24; it had grown append-heavy again
by this point — read it fresh, not as a diff against the old version).

**Read this whole file before doing anything.** If you're a fresh
session picking this up: the research is DONE. Your job is presentation
+ demo + finishing the last real engineering task (Day 20), not more
experiments.

## The one-sentence project

Can a small reasoning model use a cheap 4-sample probe to decide how to
spend its remaining test-time compute (stop / sample more / PRM-select /
abstain) better than a fixed policy? **Answer: no, not meaningfully** —
and that negative result is unusually well-evidenced (see below).

## Where things actually are: research is complete, presentation-prep phase

All four headline hypotheses have final, held-out-confirmed verdicts.
The design is frozen (`git tag design-frozen`, commit `c6c63a1`). The
held-out evaluation (Day 18) is done, one pass, no re-tuning, per
invariant #8. The full report (`report/01-*.md` through `report/08-*.md`)
is written. What's left is **Day 20** (reproducibility check + demo
rehearsal, never actually done — see "What's NOT done" below) and
turning all of this into a slide presentation + 5-6 minute demo.

**Do not re-open the research.** Three converged attempts at fixing the
controller's held-out failure already happened (2026-09-01) and found a
real, honest ceiling — see the cheat-sheet below. Further experiments
would erode the "held-out is essentially untouched" credibility that
makes every other result trustworthy, for very little additional signal.

## Headline numbers — the cheat-sheet for the presentation

| Question | Answer | Number |
|---|---|---|
| H1: does smart allocation beat a fixed policy? | **No** | dev gap ~1pp (95% CI [0.20,1.60]pp), held-out gap **0.00pp**, still 0.00pp at 4x the token budget (mentor-directed check) |
| H2: does the probe predict the winning action? | **Yes** | macro-AUROC 0.8692 (real, out-of-fold, n=754) |
| H3: does probe evidence beat a pre-hoc guess? | **Yes** | Detective 0.8692 vs. Fortune Teller 0.6479 — 4x the required margin |
| H4: does the learned controller beat fixed policies? | **No** | 0/5 budget levels on dev; **-9.30pp WORSE than best-fixed on held-out P4** |
| SELECT (PRM-based answer picking) — does it help? | **No, confirmed 6 independent ways** | oracle win rate 0.8% (6/754); PRM-argmax gets reliably WORSE than plain majority as N grows (up to -8.0pp at N=32) |
| Is G1's failure just because of the frozen token budget? | **No, directly tested** | gap identical (1.89pp) at 1024 vs. 4096 tokens, n=53 stratified sample |
| Can the held-out failure be fixed? | **Partially — tested, not speculative** | 3 independent fixes converge on the same +1.2pp recovery (-9.30pp → -8.14pp); pushing harder doesn't help further — a real ceiling |
| What's the one real positive finding? | Early-stopping value | model's majority vote stabilizes at k*≈2.17 samples; STOP alone gets 55.7% accuracy for **zero tokens** |

**If asked "do we have a product":** no — a simple 2-line heuristic
(agreement-threshold, 90.09% accuracy) ties the full learned model's
accuracy exactly. Say this plainly, don't oversell the demo.

**If asked "is this a fail":** no — a rigorous, independently-replicated
negative result (including on data never touched during development,
and after directly ruling out the obvious confound) is a successful test
of a false hypothesis, not a failed project. Real bugs were caught and
fixed along the way, and every fix was rechecked afterward, not assumed
safe.

## The two frozen, load-bearing decisions

1. **G1 FAILED** for Qwen3.5-4B. Oracle-vs-best-fixed gap ~1pp against
   an 8pp accept bar. Confirmed at full P1 scale (95% CI excludes zero),
   confirmed again on held-out (0.00pp), confirmed to be budget-
   independent. Pre-committed negative result per the brief's own §34.
2. **Action space is a true 4-class oracle label** (STOP/SAMPLE/SELECT/
   ABSTAIN), matching the brief's literal spec. SELECT's real win rate:
   0.8%, confirmed 6 independent ways (see cheat-sheet). SEARCH (the
   5th original action) was dropped via the roadmap's own G5 fallback
   after a real, unfixed continuation-strategy bug — action space stays
   4-way, this didn't further narrow it.

Neither decision has formal mentor confirmation, but both have been
discussed with the mentor since (including a mentor-directed check that
specifically supported the G1 finding). Disclosed throughout, not
hidden — see `notes/mentor-brief-summary-2026-08-24.md` and
`report/06-retrospective.md`.

## What's built and tested (real, live-verified, not just designed)

Full pipeline: `answers/`, `backends/`, `pools/`, `generation/`,
`budget/`, `evaluation/`, `selectors/`, `scoring/`, `telemetry/`,
`controller/` (real fitted `DetectiveController`, frozen at
`results/models/detective_frozen.joblib`), `replay/`, `gateway/` (full
`/solve` contract — three real outcomes, live-mode generation, anytime
budget exhaustion), `answers/thread_safety.py`. **114/115 tests
passing** (1 intentional skip). `ui/demo.py` — a working, tested
benchmark-mode CLI walkthrough (deliberately scoped down from the full
brief-described panel UI to protect report/repro time; meets the
roadmap's literal completion bar).

`search/beam.py` exists but is dropped/unused — real code, real tests,
just not part of the deployed action space. Don't resurrect it without
a real redesign of its continuation strategy (see
`report/07-next-steps.md`).

## Real data state

- **P1** (500 MATH-500, N=32): fully generated + PRM-scored. No
  logprobs (pre-dates a backend fix).
- **P2** (300 OlympiadBench-A, N=32): fully generated + PRM-scored. Has
  real per-token logprobs.
- **P4** (100 OlympiadBench-B, held-out, N=32): generated once, scored,
  evaluated. n=86/100 usable (14 filtered, ambiguous/multi-answer gold).
- **P5** (30 AIME25, held-out, N=32): generated once, scored, evaluated.
  All 30 usable. Ordinal reporting only (n too small for a CI).
- All generation ran real hours (P1: 14.5h, P2: ~13h) — every time
  estimate this project made undershot actual elapsed time by 2-4x,
  consistently. Budget accordingly if anything new needs to run.

## What's NOT done — the actual remaining work

**Day 20 has never happened.** This is the real remaining task list:

1. **`results/manifest.json` doesn't exist.** Needs building.
2. **`make reproduce-headline` has never been verified from a clean
   clone.** This is blocked on a real open decision (see next point).
3. **The `Makefile` question is unresolved.** `Makefile` is currently
   untracked (not in git — see "Git state" below), by explicit
   instruction alongside `CLAUDE.md`. But Day 20's literal completion
   bar is `make reproduce-headline` working from a clean clone — which
   is impossible without `Makefile` in the repo. **Needs a decision:
   does `Makefile` come back (it's just a build script, not the file
   with the actual disclosure concern), or does the repro check get
   verified a different way (e.g. running the underlying Python command
   directly, documented in the README)?** Ask before proceeding.
4. **The demo has never been rehearsed.** `ui/demo.py` works (spot-
   tested on several real problems across STOP/ABSTAIN outcomes;
   SAMPLE/SELECT paths are code-reviewed and reuse tested primitives
   but were never manually triggered on a live run, since those actions
   are rare — 7%/0.8% of problems).
5. **README.md** needs a real pass for "a stranger with no GPU, no API
   key" sufficiency (Day 20's own bar) — check it's actually accurate to
   current state, not just present.
6. **The slide presentation + demo script don't exist yet.** This
   session's job. See "Presentation guidance" below for what's already
   been worked out.

## Presentation guidance — already worked out, don't re-derive it

**Narrative arc that works** (not "it doesn't work, but look at it"):
question → what was built to test it (real rigor: held-out, statistics,
parity architecture) → the answer (no headroom, replicated, confound
ruled out) → the sharper insight (a good AUROC didn't guarantee good
real-world behavior — a genuinely teachable point) → the one real
capability (early-stopping) → the fix attempt (diagnosed the failure,
partially fixed it, confirmed the ceiling three ways).

**Demo structure (5-6 min), framed as explainability, not accuracy**:
1. Easy problem → STOP, correct, free — shows real capability.
2. Hopeless problem → ABSTAIN with a real machine-readable reason —
   shows honest failure instead of silent wrong answers.
3. A case that spends more budget and doesn't help — shows the actual
   research finding live, not just on a slide.
Show the confidence breakdown (`evidence.class_probs`) at each step, not
just the winning action — this is the demo's real strength given the
headline result isn't "it works."

**Have a backup recording ready.** Real, repeated transient 500/502
errors were hit tonight (`datasets-server` and the generation backend
both, independently) — a live demo failing on an unrelated network blip
in front of the mentor would be a bad, avoidable look.

## Infrastructure facts (don't rediscover these)

- **13GB RAM, 4GB swap.** Real constraint, hit once (memory-safety
  incident, Day 12). Check `free -h` before anything loading bulk pool
  data; stream one pool at a time for anything touching P2/P4/logprob-
  bearing pools, never batch-load.
- **Working directory is nested**: the real git repo is
  `~/Desktop/marginal-token-skeleton/marginal-token/` — `cd` there
  before any git command. Also: the shell's cwd has intermittently
  reset to the outer folder between tool calls this session (cause
  unclear) — always `cd` explicitly in the same command rather than
  relying on a prior `cd` having stuck.
- **Real, recurring transient network flakiness**: both the
  `datasets-server` gold-answer API and the hosted generation backend
  have thrown transient 500/502 errors multiple times this project,
  always resolved on retry. Add retry logic to any new network-calling
  script rather than assume a clean first attempt.
- **Hosted endpoints**: `policy_primary` (Qwen3.5-4B via a third-party
  unsloth GGUF conversion, NOT official weights — disclosed to the
  mentor and in `report/05-discussion.md`), `prm_primary`
  (Qwen2.5-Math-PRM-7B, zero failures across thousands of calls),
  `policy_secondary_stretch` (2B, used once diagnostically). API key:
  `.env` → `HOSTED_ENDPOINT_API_KEY` (gitignored). Always
  `set -a && source .env && set +a` in the SAME command as any script
  that needs it — env vars don't persist across tool calls.
- **Raw GPU SSH** (`root@194.149.138.21`) has no spare capacity (98%
  VRAM on standing services). `.12` has no working credential. Not a
  path to the official (non-GGUF) weights without new infrastructure.

## Git state

**Clean as of commit `d9ab977`, tag `design-frozen` on `c6c63a1`
(earlier commit — the tag was never moved, which is correct, it marks
when the design froze, not the latest commit).** Both pushed and
verified against `origin/main`. **`CLAUDE.md` and `Makefile` are
deliberately untracked** (present locally, not committed) — an earlier
commit briefly included both by mistake, caught and corrected via a
contained rewrite of just that one commit (never touched the separately-
pushed, differently-authored `f59ae65`, which has a stale commit
message but real, correct content — left alone deliberately, not
rewritten). Verify with `git status --porcelain` before any new commit
that only `CLAUDE.md`/`Makefile` show as untracked, nothing else
unexpected.

Before adding `Makefile` back (if that's the Day-20 decision): it was
never actually changed from the original, so re-adding it is just
`git add Makefile` — no content decision needed, just the disclosure
question from "What's NOT done" above.

## Read order for a fresh session

1. `CLAUDE.md` (short, mostly stable contract — the frozen invariants
   are still real, though the "work through the roadmap" framing is
   now stale; this file's "Current phase" section says what actually
   matters right now).
2. This file, in full.
3. `report/04-results.md` for the complete, presentation-ready results
   writeup, and `report/06-retrospective.md` for the honest what-went-
   well/what-went-wrong.
4. `notes/2026-09-01.md` for the controller-fix exploration specifically
   (most recent, most relevant to "can we improve this" questions).
5. Dated logs (`notes/2026-08-2*.md`) only for evidence-trail detail
   behind a specific claim, not to figure out current state.
