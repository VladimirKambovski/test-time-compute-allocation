# Project Start vs. Project End, Planned vs. Actual Timeline

## Project start vs. project end

| | At project start | By project end |
|---|---|---|
| Action space | 5 actions (STOP/SAMPLE/SELECT/SEARCH/ABSTAIN), all in scope | 4 actions in the frozen label space and the deployed controller; SEARCH dropped via the roadmap's own documented G5 fallback after a real, unrecovered continuation-strategy bug |
| N floor | N=64 (later N=128 SHOULD-extension) | N=32 MUST floor, changed by explicit instruction on 2026-08-20 after real generation-time data made N=64 impractical on schedule; every headline result computable at N=32, nested prefixes support any smaller budget |
| Central hypothesis expectation | Open question — does per-query allocation choice matter enough to predict and act on | Answered, negatively, with unusually strong support: ~1pp gap on dev, 0.00pp on held-out, 0.00pp again at 4x the token budget. Real headroom essentially does not exist for this policy at this difficulty range. |
| SELECT (PRM-weighted selection) | Expected to be a live, real controller action | Measured at 0.8% win rate, confirmed 6 independent ways; still tracked as a real oracle-label class (not silently excluded) per the frozen spec's literal 4-class design |
| Held-out generalization | Unknown until Day 18 | Directly tested and confirmed — the dev-side negative findings replicate exactly on data never touched during development, which is stronger validity evidence than most projects at this scale get |
| Demo scope | Full panel UI (probe/evidence/controller/spend/outcome, comparison mode) | Deliberately scoped to a benchmark-mode CLI walkthrough that meets the roadmap's literal completion bar, trading UI polish for report/reproducibility time under a compressed deadline |
| Model provenance | Assumed official HF weights | Discovered Day 3, disclosed here: served via a third-party GGUF conversion, not the official safetensors — flagged as a caveat on every result, not found to have produced an obvious artifact but genuinely untested against the official checkpoint |
| Process discipline | Aspirational — "log every decision, disclose every negative result" | Held under real pressure: a real git-history mistake (CLAUDE.md/Makefile briefly committed), a real memory-safety incident, and a real methodology bug (duplicate-pool double-counting) were all caught, fixed, and disclosed rather than quietly corrected or omitted |

## Planned vs. actual timeline

The original plan (`docs/brief.md` section 32) budgeted ~146 working hours
over 20 days (~7.3h/day), with two explicit background-generation windows
(P1 during Days 6-8, P2 during Days 9-10) running unattended alongside
active work.

**What's precisely measured:** the two large background generation runs,
which have exact logged durations against the plan's own implicit
estimate (a combined 4-7h P1+P2 estimate, stated in `notes/2026-08-23.md`
as the a-priori figure being compared against):

| Run | Estimated (combined a-priori) | Actual |
|---|---|---|
| P1 (500 problems, N=32) | part of 4-7h combined | **14.5h** |
| P2 (300 problems, N=32) | part of 4-7h combined | **~13h** |

Every subsequent background generation job in this project (the max_tokens
ablations, the held-out P4+P5 generation, the mentor-directed cross-band
check) showed the same pattern to a smaller degree — actual duration
consistently at or above the estimate given before launching, never
meaningfully under.

**What's honestly reported at lower precision:** Days 13-19's work
(supporting DIAGNOSE analyses through report finalization, ~58 planned
hours across 7 originally-separate work days) was executed in a single,
much shorter, continuous real-calendar window under external deadline
pressure — not tracked hour-by-hour against the original per-day budget.
Rather than invent a false-precision hour count for this stretch, the
honest summary is qualitative: the *planned* 4-week/20-day cadence
compressed into a real calendar span closer to 4-5 days once the deadline
was set, with Days 13-19's analysis, engineering, and writing work
happening in one extended push rather than 7 separate ~7-8 hour days, and
several long unattended background jobs (max_tokens ablations, held-out
generation, PRM scoring) run overnight or in parallel with active work
rather than sequentially.

**Major deviations and their explanations:**

1. **Every background generation run undershooting its time estimate by
   2-4x** — the single most consistent pattern in this project's real
   data, present from Day 4's original 100-problem pool through the final
   held-out generation on Day 18. No single explanation fully accounts for
   it; the hosted backend's real per-sample throughput was simply slower
   than every a-priori estimate assumed, consistently, across many
   different problem sets and decode configs.
2. **The search arm (A3) consumed real Day 12-13 engineering time and was
   then dropped**, rather than completing as originally scoped — a real,
   disclosed scope reduction via the roadmap's own G5 fallback, not a
   silent narrowing of the research question.
3. **Days 13-19 compressed from ~7 planned work days into a much shorter
   real window** once an external 3-4 day deadline was set partway
   through the project — the single largest deviation from the original
   20-day cadence, and the reason several SHOULD-tier and one MUST-
   adjacent item (listed in full in `07-next-steps.md`) were deliberately
   cut rather than completed.
4. **A held-out result stronger and more specific than the plan
   anticipated.** The original plan expected Day 18 to simply confirm or
   fail to confirm the dev-side findings; it did both — confirming the
   dev-side negative results exactly, and additionally surfacing a new,
   more consequential result (the learned controller actively
   underperforming a trivial baseline on held-out) that the dev-side
   analysis alone would not have caught.
