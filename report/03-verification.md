# Literature Claim Verification (V1-V5)

_Skeleton drafted Day 2 (was due Day 1 per docs/brief.md §32 — caught late,
backfilled here). Verdicts filled in as each claim is actually tested;
"reproduced / partially / not reproduced / out of scope" per docs/brief.md
§9._

| # | Claim | Source | Test | Verdict |
|---|---|---|---|---|
| V1 | A 1B model with compute-optimal TTS beats a 405B model | 2502.06703 | Check scope: MATH-500-only, N=512, PRM-dependent; paper itself reports underperformance on AIME24. Does gap-closing survive on OlympiadBench at N=64? | _pending — E2/E7 (Week 3)_ |
| V2 | PRMs beat majority voting | 2501.07301 | Reproduce the margin (B3); does it clear the paired bootstrap CI? | _pending — Day 5 (B3 reproduction)_ |
| V3 | Self-consistency plateaus, can decline after peak on hard items | 2508.00410 | Locate plateau-N per benchmark and difficulty band | _pending — Day 4 (maj@k curve) / E4 (Week 3)_ |
| V4 | PRM argmax winners are length-biased | 2606.09078 | Regress winner length and step count against pool median | _pending — E4 (Day 13)_ |
| V5 | Cheap agreement signals give large savings at ~zero accuracy cost | 2305.11860 | Reimplemented as a predictor comparator (fixed agreement threshold); measured against H2 | _pending — E5 (Day 14)_ |

Related but not part of the V1-V5 set (tracked in `report/01-literature.md`
instead, since they inform novelty/gap rather than a specific reproducible
claim): 2604.17433, 2606.08098, 2607.08065, 2506.12721, 2604.14853.
