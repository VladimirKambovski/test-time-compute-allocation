# Predicting Where the Next Token Should Go

Evidence-driven test-time compute allocation for small reasoning
models, delivered as a Compute-Aware Reasoning Gateway.

**If you are Claude Code:** read `CLAUDE.md` first, then
`notes/HANDOFF.md` for current state — the research is complete, this
is now presentation/reproducibility-finishing work, not active
experimentation.

**If you are a human, new to this project:** start with `report/`, not
`docs/brief.md` — it's the actual finished writeup, in order:
`01-literature.md` (the question and prior work) through
`08-start-vs-end.md` (what changed along the way). `docs/brief.md` is
the original frozen specification, useful for methodology detail, not
the place to start.

## The finding, in one paragraph

Can a small reasoning model use a cheap 4-sample probe of its own
answers to decide how to spend the rest of its compute budget — stop,
sample more, use a process-reward-model to pick the best answer, or
decline — better than a fixed policy? **No, not meaningfully** — the
gap between the best possible allocator and the best fixed policy is
close to zero, on two development benchmarks and, independently, on
held-out data the system never touched during development. The
predictor itself is genuinely good at telling these cases apart
(AUROC 0.87), but on held-out data actually deploying it would have
cost accuracy rather than saved it — a real, diagnosed, partially-
fixable calibration problem, not a flaw in the underlying signal. Full
detail and honest limitations: `report/04-results.md` and
`report/05-discussion.md`.

## Quickstart

```bash
uv sync
make test                  # runs the invariant test suite
make reproduce-headline    # rebuilds all headline tables/figures from
                            # committed cached artifacts -- no GPU, no
                            # API key required
```

`make reproduce-headline` reads only committed, cached artifacts
(`results/figures/`, `results/heldout_results.json`, the frozen model at
`results/models/`) — it does not call any model or PRM endpoint. If
`make` isn't available in your environment, the underlying commands are
plain `python -m marginal_token.*` module invocations — see `Makefile`
for the exact ones.

## See the demo

```bash
python ui/demo.py --random          # walks a random dev problem through
                                     # the full decision path, using only
                                     # cached data -- zero API calls
python ui/demo.py test/algebra/2176.json   # a specific problem
```

## Structure

```
CLAUDE.md              contract for AI-assisted development on this repo (local only, see below)
docs/brief.md           full frozen specification (source of truth for methodology)
docs/roadmap.md          day-by-day task list (historical -- the research is complete)
notes/                    dated decision + experiment log, plus notes/HANDOFF.md (current state)
configs/                  one YAML per policy / PRM / benchmark / pool / experiment
src/marginal_token/        the system, one package per responsibility
tests/                     invariant tests, 114/115 passing
results/                   figures, held-out results, the frozen predictor model
report/                    the finished report, 8 sections
ui/demo.py                 benchmark-mode CLI demo (cached artifacts only)
```

## Status

Research complete. Design frozen (`git tag design-frozen`). Held-out
evaluation done, one pass, per the project's own no-re-tuning rule.
Primary policy model: Qwen3.5-4B (served via a third-party GGUF
conversion — disclosed limitation, see `report/05-discussion.md`).
Primary verifier: Qwen2.5-Math-PRM-7B.

**A note on `CLAUDE.md` and `Makefile`**: if you cloned this repo and
don't see a `CLAUDE.md`, that's deliberate — it's kept local-only by
project choice, not missing by accident. `Makefile` should be present;
if it's not, `make` commands above won't work — see `notes/HANDOFF.md`
for why and what to do about it.
