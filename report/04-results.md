# Results

## Headline hypothesis verdicts

| Hypothesis | Question | Verdict | Evidence |
|---|---|---|---|
| H1 | Does the allocation choice matter, and is it heterogeneous across queries? | **REJECT** (no meaningful headroom) | G1: oracle-vs-best-fixed gap ~1pp on dev (95% CI [0.20, 1.60]pp, full P1 scale), 0.00pp on held-out P4, 0.00pp again at 4x the token budget |
| H2 | Can cheap probe evidence predict the winning action? | **ACCEPT** | Detective macro-AUROC 0.8692 (real, out-of-fold, grouped 5-fold CV, n=754) — comfortably clears the 0.70 accept bar |
| H3 | Does post-hoc probe evidence beat a pre-hoc query-text guess? | **ACCEPT** | Detective (0.8692) beats Fortune Teller (0.6479) by 0.2213 — over 4x the required 0.05 margin |
| H4 | Does acting on the prediction improve the quality-compute frontier? | **REJECT** | Detective beats the best fixed policy at 0/5 budget levels on dev; on held-out P4 it scores 9.3pp *worse* than the best fixed policy |

## H1: the action-value landscape

**Gate G1 (oracle-vs-best-fixed-policy gap ≥8pp to accept) failed on every
check run**, at every scale tested: MATH-500 N=32 (1pp), N=64 (2pp),
OlympiadBench-A (0pp exactly), and finally the full 500-problem P1 pool
(95% CI [0.20, 1.60]pp — genuinely excludes zero, but the point estimate
never moved). This is logged as the project's pre-committed negative
result, per the brief's own protocol for handling a failed primary gate.

**Mechanism.** The model's running majority vote locks onto its final
answer almost immediately for most problems — mean stabilization point
k*≈2.17 samples out of 32 (Day 4). This explains *why* the gap is small:
STOP and SAMPLE mostly agree, leaving an oracle little room to improve on
either. This same finding is the project's real positive result — see
"Cost-efficiency" below.

**Difficulty-band structure, and a real correction to it.** The aggregate
~1pp gap is not uniform: on the full P1∪P2 landscape (754 problems, 5
difficulty bands), the hardest 20% initially appeared to have a 0.000
ceiling for every action including the oracle, and one moderately-hard
band appeared to carry ~4pp of gap on its own. **Investigation (prompted
by an unexpectedly large gap on a temperature-ablation subset) found this
was largely a token-budget artifact, not a clean reasoning-difficulty
signal.** The oracle-label majority computation silently excludes failed
extractions from the vote; on the hardest band, 99% of problems averaged
only 0.21/32 successful extractions at the frozen 1024-token budget — the
model was being cut off mid-derivation, not failing to reason. Demonstrated
directly, not just inferred: regenerating a systematic n=30 sample of the
hardest band's problems at 4096 tokens flipped 25/30 (83.3%) from
"unwinnable" to a real, mostly-cheap-to-resolve action. **The honest,
complete characterization is a mix**: roughly 5/6 of that population
recovers with more budget, roughly 1/6 has a real ceiling beyond
truncation (stayed dead even at 4x tokens).

**This correction does not change H1's aggregate verdict.** A follow-up,
mentor-directed check (stratified sample across all 5 difficulty bands,
n=53, canonical enumeration) found the gap is *identical* at 1024 and 4096
tokens — 1.89pp both times (95% CI [0.00, 5.66]pp both times; paired
change-in-gap test 95% CI [-7.55, 3.77]pp, clearly includes zero). Raw
accuracy roughly doubles at the larger budget (35.9%→90.6% oracle
accuracy) — the recovery effect is real — but the *relative* headroom for
smart allocation over the best fixed policy does not move. This directly
rules out "the frozen token budget explains G1's failure" as an
alternative explanation, rather than leaving it untested.

## SELECT: confirmed rare 5 independent ways

The oracle action distribution over the full 754-problem dev set is STOP
56%, ABSTAIN 36%, SAMPLE 7%, **SELECT 0.8%** (6/754). Over a third of all
problems are unwinnable by any strategy at the frozen budget (ABSTAIN is
oracle-correct only because nothing else works either); SELECT — PRM-
weighted answer selection — is a real, measured 4th class, but wins almost
never. Five independent checks converge on this:

1. **Oracle ceiling on the primary policy** (4B): 0.8% (this measurement).
2. **A weaker-policy diagnostic** (2B): SELECT's rate stays near-zero.
3. **A real PRM reproduction** (B3, Day 5): PRM-weighted majority equals
   plain majority exactly, zero individual-problem flips at full P1 scale.
4. **Temperature ablation, targeted at the 4 known SELECT-oracle
   problems**: the one clean, uncontaminated case is completely stable
   across temperature (SELECT wins identically at 0.8 and 1.0); the other
   3 (extraction-confounded) just relabel noisily based on which few
   samples happened to complete — no case where raising temperature
   created or strengthened a real SELECT win.
5. **max_tokens ablation, same 2 confounded SELECT cases**: both flip to
   STOP with perfect 32/32 extraction once given room to finish — SELECT's
   apparent win vanishes entirely rather than strengthening.

The E4 crossover analysis adds a sixth, sharper angle: even the simplest
possible PRM-based selector (always pick the single highest-scored sample,
no voting at all) is reliably *worse* than plain majority, and gets
increasingly worse as more samples become available (significant negative
gap at every budget level from N=4, widening to -8.0pp at N=32). PRM-based
answer selection does not just fail to add value in this data — the
simplest form of it actively subtracts value, with the deficit growing the
more compute you give it.

## H2/H3: the predictor

Detective (multinomial logistic regression on the 4-sample probe,
grouped-by-problem/stratified-by-benchmark 5-fold CV, n=754):

| class | AUROC | precision | recall | support |
|---|---|---|---|---|
| STOP | 0.9955 | 0.986 | 0.998 | 420 |
| ABSTAIN | 0.9598 | 0.820 | 0.961 | 273 |
| SAMPLE | 0.8642 | 0.478 | 0.162 | 55 |
| SELECT | 0.6994 | 0.000 | 0.000 | 6 |

Macro-AUROC 0.8692, overall accuracy 92.4% vs. 55.7% for a trivial
"always guess the majority class" baseline — a real 36.7pp lift, not
noise. **The honest, narrower reading of the imbalance concern**: SAMPLE's
AUROC is genuinely good (0.86 — real discriminative signal exists), but
its hard-decision (argmax) recall is only 16.2%, because the class priors
(420 STOP vs. 55 SAMPLE) mean close calls default to the dominant classes.
This is a calibration issue, not "no signal." A `class_weight='balanced'`
refit was tested as a fix: real trade-off, not a clean win — SAMPLE recall
rises to 55.9% but ABSTAIN recall drops from 96.1% to 65.7% and overall
accuracy drops from 90.1% to 77.6%. The plain model is reported as the
headline result; the balanced variant is documented as an honest ablation.
SELECT's 0% recall is structural (6 examples total, ~1 per CV fold) and
unaffected by reweighting either way.

Fortune Teller (the pre-hoc query-embedding control, non-negotiable for
H3's falsifiability): macro-AUROC 0.6479. Real signal in the query text
alone — harder-looking problems predict worse outcomes even before
generation — but Detective's post-hoc probe evidence adds a real,
substantial 0.2213 (over 4x the required 0.05 margin) on top of it.

A simple fixed-agreement-threshold comparator (predict STOP if the probe's
top-answer share ≥0.25, else ABSTAIN) ties Detective's overall accuracy
exactly (both 90.09%). Detective's real advantage over this dumb 2-bucket
rule shows up specifically in macro-AUROC/class-level discrimination
(only Detective can signal SAMPLE at all), not in raw accuracy.

## H4 and the cost frontier

E7 (7 policies × 5 matched-token budgets {2,4,8,16,32}, n=754, real
out-of-fold Detective/Fortune Teller predictions):

| policy | B=2 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| Miser (always STOP) | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 |
| Spendthrift (always SAMPLE) | 0.499 | 0.557 | 0.580 | 0.613 | 0.629 |
| UniformSelect (always SELECT) | 0.455 | 0.513 | 0.558 | 0.581 | 0.614 |
| Gambler (random) | 0.528 | 0.557 | 0.570 | 0.594 | 0.602 |
| Oracle (true ceiling) | 0.558 | 0.558 | 0.581 | 0.614 | 0.633 |
| Fortune Teller (pre-hoc) | 0.474 | 0.474 | 0.474 | 0.474 | 0.474 |
| **Detective (learned)** | 0.557 | 0.557 | 0.561 | 0.562 | 0.566 |

Detective never beats the best fixed policy at any budget level (0/5 —
gate G7's own bar for accepting H4 is ≥3/5). E8 cost accounting: Miser
achieves 55.7% accuracy for **zero tokens**; Spendthrift spends ~26,577
tokens/problem for only 7.2pp more (62.9%) — diminishing returns,
quantified. SAMPLE beats SELECT on cost-per-correct-answer too (42,276.7
vs. 43,289.4 tokens) — SAMPLE strictly dominates SELECT on this data,
cheaper and more accurate both.

## Held-out evaluation (Day 18, one pass, no re-tuning)

The **frozen** Detective model (fit once on the full dev set, never
refit) against P4 (OlympiadBench-B, in-distribution) and P5 (AIME25,
out-of-distribution):

**P4 (n=86/100, 14 filtered for ambiguous/multi-answer gold, same rate as
P2's dev filtering):**

| | value |
|---|---|
| STOP accuracy | 0.3023 |
| SAMPLE accuracy | 0.4186 |
| SELECT ceiling | 0.4186 |
| Oracle ceiling | 0.4186 |
| Best fixed policy | 0.4186 |
| **G1-style gap** | **0.00pp** |
| **Detective's real achieved accuracy** | **0.3256** |
| **Detective vs. best fixed** | **-9.30pp** |

Both the dev-side G1 finding (0.00pp gap) and SELECT's zero added value
(SELECT ceiling = SAMPLE accuracy exactly) **independently replicate on
data never touched during development** — strong evidence the findings
are real, not dev-set artifacts. The new, more consequential result:
**Detective doesn't just fail to help on held-out — it measurably hurts**,
scoring 9.3pp below simply always sampling. Traced to Detective
under-using SAMPLE relative to the true optimal action distribution on
this data (predicted action counts: abstain=54, stop=27, sample=5; true
optimal: abstain=50, sample=10, stop=26) — it abstains on some problems
that SAMPLE would have solved.

**P5 (n=30/30, ordinal reporting only per this benchmark's own
`statistics: ordinal_only` designation — no CI):**

STOP = SAMPLE = SELECT = oracle = Detective = **6.67% (2/30)**. There is
essentially no headroom for any strategy on AIME-difficulty problems at
this policy size and budget — a genuine capability ceiling of the 4B
model, not a controller failure. Detective's action distribution (2 stop,
28 abstain) matches the true optimal distribution exactly, though there
was very little room to be right or wrong either way.
