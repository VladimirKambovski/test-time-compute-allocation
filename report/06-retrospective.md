# What Went Well / What Went Wrong

## What went well

**Negative results were treated as real deliverables, not failures, from
the start** — per the brief's own §34 framing, and it held up in practice.
G1 failing on Day 4 didn't stall the project; it sharpened the research
question into something more precise and better-evidenced by the end (a
tiny-but-real gap, replicated on held-out data, robust to a 4x token
budget change) than a vague "adaptive routing probably helps a bit" claim
would have been.

**Real bugs were caught before they propagated, repeatedly, because
results were verified rather than trusted on sight.** Four `math_verify`
usage bugs (Day 3), a tie-handling inconsistency (Day 5→8), a threading
crash found twice in two different code paths (Day 6, then again in the
live gateway on Day 10), a missing-logprobs bug (Day 10), a `.gitignore`
path mismatch that would have committed 17GB of pool data, a duplicate-
pool double-counting bug (caught by noticing an unexplained problem-count
discrepancy rather than trusting a script's first output), and a
demo-script answer-stringification bug that silently reported correct
answers as wrong (caught by manually verifying one suspicious-looking
result against the already-tested oracle-label logic before trusting it).
In every case, the pattern was the same: something looked slightly off,
and that feeling was investigated rather than rationalized away.

**The parity architecture (one Controller, two consumers) held up under
real pressure.** `test_controller_parity.py` never had to be weakened or
skipped, even while the gateway's `/solve` contract, the demo, and the
frozen model artifact were all being built under significant time
pressure late in the project.

**Decisions made without waiting for confirmation were disclosed, not
hidden, consistently.** G1's verdict, the SELECT-narrowing call, the
search-arm drop, the freeze execution, and the held-out generation
launch were all made and acted on before mentor sign-off arrived, given
real time constraints — but every one is logged with full reasoning in
`notes/`, flagged as revisable, and was actually revisited once mentor
input did arrive (the cross-band max_tokens check was a direct, specific
response to a mentor suggestion, run and reported honestly regardless of
which way it came out).

## What went wrong

**Every real time estimate this project made undershot actual elapsed
time by 2-4x, consistently, not as a one-off.** P1 (500 problems, N=32)
took 14.5 hours against a combined P1+P2 a-priori estimate of 4-7 hours.
P2 took ~13 hours. The original 20-day/146-hour plan compressed into a
much shorter real calendar window near the end, driven by external
deadline pressure rather than the original schedule — see
`08-start-vs-end.md` for the full planned-vs-actual table.

**The search arm (A3, bounded PRM-guided beam search) was dropped after
real, unrecovered engineering difficulty.** The continuation-prompting
design made the model repeat its opening sentence instead of progressing,
confirmed on all 3 smoke-test problems, never reaching a real boxed answer
in 8 rounds on any of them. The roadmap's own G5 fallback ("if broken,
drop A3 and report as a negative engineering result") was invoked — one
day early relative to the original schedule, given the compressed
timeline — rather than sinking more unscheduled time into a redesign.
This is a real scope reduction, honestly reported, not concealed inside a
narrower research question.

**One real, disclosed process incident: a commit briefly included files
that should not have been pushed.** `CLAUDE.md` and `Makefile` were
included in an early version of the design-freeze commit, contrary to
explicit instruction; caught immediately, and corrected via a contained
rewrite of only that one commit (the pushed history was not otherwise
touched — a separate, earlier commit with an unrelated message/content
mismatch was deliberately left alone, since rewriting pushed history over
a cosmetic discrepancy carries more risk than it resolves). Verified clean
on the remote afterward. Disclosed here in the same spirit as every other
finding in this project — a real mistake, caught fast, fixed, not hidden.

**A real, memory-safety-relevant infrastructure incident.** An early
version of a local analysis script batch-loaded both dev pools into
memory at once; harmless for P1 (no logprobs) but P2's real per-token
logprob arrays pushed usage to 9.4GB+ on this project's 13GB machine and
filled all available swap, caught live when the machine visibly started
struggling. Rewritten to stream one pool at a time; the standing rule
("never batch-load a logprob-bearing pool across many problems") held for
the rest of the project without a repeat incident.

**A methodology bug (duplicate pool directories inflating aggregate
counts) went undetected for several analysis runs before being caught.**
The root cause — some problems have more than one valid on-disk pool
directory, plausibly from historical retry attempts — was structurally
possible from early in the project but only became consequential once
aggregate scripts started enumerating "all problems" via a directory
glob rather than canonical content-addressed lookup. Once found, every
affected script was rerun with the fix; the headline verdicts were
unaffected, but this was closer to a real correctness risk than the
project's other caught-and-fixed bugs, and is the clearest argument in
this project's own experience for "verify aggregate counts against a
known-good reference before trusting them," which is now the standing
practice for any future aggregate analysis on this codebase.

**Several SHOULD-tier and one MUST-adjacent supporting analysis were cut
under time pressure rather than completed.** N=64 pool extension, the
full brief-described demo UI (probe/evidence/controller/spend panels,
comparison mode — a working benchmark-mode CLI walkthrough was built
instead), a 30-trace qualitative inspection of PRM-argmax misses, and the
per-difficulty-band component of V3's plateau/decline check. Each is
listed explicitly in `07-next-steps.md` rather than silently dropped from
the record.
