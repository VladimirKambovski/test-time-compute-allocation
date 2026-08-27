# Next Steps

## Directly scoped-out this session, worth real follow-up

- **V3's per-difficulty-band plateau/decline check.** The aggregate
  stabilization evidence is solid; the specific claim (accuracy can
  *decline* after a peak specifically on hard items) was never isolated
  by band. Cheap to run — pure analysis over already-cached data, similar
  cost to E4 — a natural first item for anyone picking this project back
  up.
- **A systematic max_tokens sweep across the rest of band 0 and band 1.**
  This project's max_tokens findings are demonstrated on real, generated
  data (n=30 for band 0 specifically, n=53 for the cross-band check), not
  just inferred — but band 0 alone has ~151 problems in the full P1∪P2
  landscape, and only a fraction have been directly tested at the larger
  budget. A full sweep would sharpen the "5/6 recovers, 1/6 has a real
  ceiling" split into a precise, well-powered number rather than a
  small-sample estimate.
- **A redesigned continuation strategy for the search arm (A3).** The
  specific failure mode found (the model repeats its opening sentence
  instead of progressing) is a real, diagnosable prompt-engineering
  problem, not evidence that bounded PRM-guided search is fundamentally
  unworkable for this policy. A genuinely different continuation format
  (e.g. explicit step-numbering continuation cues, or a different
  prompt structure entirely) is worth a real, supervised design pass
  before concluding search doesn't help here.
- **A real, measured PRM-forward-cost figure**, replacing the disclosed
  "assume it costs the same as one sample" convention used throughout the
  cost-accounting numbers in this report. This would sharpen E8's
  cost-per-correct-answer comparisons without changing their qualitative
  conclusion (SAMPLE already beats SELECT on cost given SELECT's
  near-zero win rate, so a different assumption would move the exact
  numbers, not the ranking).
- **The 30-trace qualitative inspection of PRM-argmax misses**, originally
  scoped for Day 13 and cut in favor of the cheaper, higher-leverage E4
  quantitative check. Would add concrete illustrative examples for the
  report/demo, complementing (not replacing) E4's statistical result.

## Requires real infrastructure or scope decisions beyond this project

- **Reproduce on the official Hugging Face safetensors weights**, not the
  third-party GGUF conversion every number in this report currently comes
  from. Would directly test whether the disclosed provenance caveat in
  `05-discussion.md` actually matters for any of the headline findings.
- **A class-weighted or cost-sensitive production variant of Detective**,
  informed by a real deployment cost function for a wrong ABSTAIN vs. a
  missed SAMPLE opportunity — this project measured the trade-off
  (`class_weight='balanced'`) but has no principled basis for choosing a
  point on it without knowing the actual downstream cost asymmetry.
- **N=64 pool extension.** Explicitly SHOULD-tier and conditioned on
  being on schedule throughout the brief; the project was not, and it was
  skipped by deliberate choice, not oversight. Would sharpen the
  crossover/plateau analyses with a 6th budget level.
- **The full brief-described demo UI** (probe/evidence/controller/spend/
  outcome panels, comparison mode) — a working benchmark-mode CLI
  walkthrough exists and meets the roadmap's literal completion bar
  ("someone can pick a problem and watch the full decision path to an
  outcome"), but the richer visual panel design in the original brief was
  deliberately not built, in favor of protecting time for the report and
  reproducibility check.
- **Testing whether the core negative finding (little allocation headroom)
  is specific to this policy size**, using the weaker 2B policy already
  available on the same backend at full scale, not just as the
  diagnostic-scale check this project ran. A genuinely larger or smaller
  policy might show a real, different action-value landscape — this
  project's finding is scoped to one ~4-5B-parameter model, not a general
  claim about test-time compute allocation.
