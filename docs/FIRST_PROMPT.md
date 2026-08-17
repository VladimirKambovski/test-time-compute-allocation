# First message to paste into Claude Code

This is not documentation — it's the literal text to paste as your
opening message once you `cd` into the repo and start Claude Code.

---

```
Read CLAUDE.md in full, then read docs/roadmap.md's Day 1 section.

Do not write any implementation code yet. Before touching code:

1. Confirm you understand the frozen model choices (Qwen3.5-4B primary
   policy, Qwen2.5-Math-PRM-7B primary PRM) and the five non-negotiable
   invariants in CLAUDE.md. Summarize them back to me in your own words
   so I can confirm you've internalized them, not just skimmed them.

2. Start Day 1 of docs/roadmap.md: the novelty check on the five papers
   listed in docs/brief.md section 1, and gate G0 (verifying Qwen3.5-4B
   is servable on at least two backends). For G0 specifically: search
   for the exact, correct Hugging Face repo slug for Qwen3.5-4B --
   docs say "VERIFY exact HF slug" in configs/policies/qwen3.5-4b.yaml,
   and I have not independently confirmed it. Do not guess -- find the
   real one and update the config file.

3. Log everything in notes/2026-XX-XX.md using the template in
   notes/TEMPLATE.md (copy it, don't edit the template itself).

Ask me before making any decision that isn't already specified in
docs/brief.md or CLAUDE.md. If a gate fails, follow its documented
fallback -- don't improvise a different one without telling me first.

Work only on Day 1. Stop and report back when Day 1's "Done when"
condition is met, even if you could technically keep going.
```

---

## Notes on using this well

- **Do this literally day by day for the first week or two.** Once you
  trust the rhythm (it reads the roadmap, does the day, logs to notes/,
  stops), you can say "continue to the next day" instead of re-pasting
  the whole thing.
- **Re-paste the CLAUDE.md read instruction if you ever start a fresh
  Claude Code session** (new terminal, new day, whatever) — context
  doesn't carry over between sessions automatically.
- **If Claude Code ever produces something that contradicts CLAUDE.md's
  invariants**, that's a bug in the session, not a reason to update the
  invariant. Point it back at CLAUDE.md rather than approving the drift.
- **Gate failures are normal, not emergencies.** If G0 or G1 fails on
  Day 1 or Day 4, that's the process working correctly — the roadmap
  already has the fallback written down. Let it follow the documented
  path rather than treating it as something to escalate to me first
  (except G10, which is explicitly a "stop and ask" gate by design).
