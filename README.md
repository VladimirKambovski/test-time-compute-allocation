# Predicting Where the Next Token Should Go

Evidence-driven test-time compute allocation for small reasoning
models, delivered as a Compute-Aware Reasoning Gateway.

**If you are Claude Code:** read `CLAUDE.md` first. It is the contract
for this repo. Then `docs/roadmap.md` for what to work on today.

**If you are a human:** `docs/brief.md` is the full frozen research
specification. This README is intentionally short.

## Quickstart

```bash
uv sync
make test                  # runs the invariant test suite
make reproduce-headline    # rebuilds all headline tables/figures from
                            # committed cached artifacts -- no GPU, no
                            # API key required
```

## Structure

```
CLAUDE.md            contract for AI-assisted development on this repo
docs/brief.md         full frozen specification (source of truth)
docs/roadmap.md        day-by-day task list
notes/                 dated decision + experiment log (append-only)
configs/               one YAML per policy / PRM / benchmark / pool / experiment
src/marginal_token/     the system, one package per responsibility
tests/                  invariant tests -- see CLAUDE.md for what each guards
results/                figures, tables, manifest.json (generated only)
report/                 incremental report sections
ui/                     the demo
```

## Status

Frozen scope, subject to gates G0–G10 defined in `docs/brief.md`.
Primary policy model: Qwen3.5-4B. Primary verifier: Qwen2.5-Math-PRM-7B.
