"""
Bounded PRM-guided beam search (A3/SEARCH), per docs/brief.md's design
(line 58; §17 E3; Day 12 roadmap item). Cannot be cached -- the
verifier steers decoding, so every configuration is a fresh live run.

**Continuation design decision, stated explicitly, not assumed silently:**
this hosted backend only exposes OpenAI-compatible CHAT completions (no
raw-completion "continue this exact prefix" endpoint). Beam
continuation is therefore implemented by re-prompting: each step, the
model is shown the original problem PLUS the accumulated partial
solution so far, and asked to continue from there, stopping at the next
step boundary via `stop_sequences=("\\n\\n",)` (the same double-newline
convention already validated for segmentation, Day 5's G3 gate). This is
a standard pattern for step-wise guided decoding against chat-only APIs;
flagged here because it's a real design choice with its own failure
modes (the model could ignore the "don't repeat" instruction), not
something to trust without checking real output.

**Budget discipline (invariant #4):** every token generated, kept OR
discarded beam alike, is charged, plus every PRM forward -- see
`charge_search_action` in `budget/accounting.py`. Undercounting either
would invalidate every matched-token comparison this arm feeds into.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from marginal_token.answers.extraction import extract_answer, find_boxed_spans
from marginal_token.answers.taxonomy import FailureStatus
from marginal_token.backends.base import Backend, DecodeConfig
from marginal_token.budget.accounting import Charge, charge_search_action
from marginal_token.scoring.prm_client import PRMClient
from marginal_token.scoring.segmentation import segment_double_newline

STEP_STOP = "\n\n"
CONTINUATION_TEMPLATE = (
    "{problem}\n\n"
    "Partial solution so far (do not repeat it -- continue directly "
    "from where it stops):\n{prefix}"
)


@dataclass
class Beam:
    text: str = ""  # accumulated generated text so far
    tokens_spent: int = 0  # policy tokens spent on THIS beam's surviving history
    finished: bool = False  # a valid boxed answer was found (checked via extract_answer, not a substring guess)
    mean_reward: float = 0.0  # last known PRM score (0.0 until first scored -- never treated as "confirmed good")


@dataclass
class SearchResult:
    final_text: str
    charge: Charge
    n_steps: int
    beams_history: list[list[Beam]] = field(default_factory=list)  # for inspection/debugging, not required downstream


def _build_prompt(problem: str, beam: Beam) -> str:
    if not beam.text:
        return problem
    return CONTINUATION_TEMPLATE.format(problem=problem, prefix=beam.text)


def _is_finished(text: str) -> bool:
    """A beam is finished when its accumulated text contains an actual
    `\\boxed{...}` span that extracts cleanly.

    Real bug found live during the Day-12 smoke test, not hypothetical:
    an earlier version called `extract_answer(text, finish_reason="stop")`
    directly, which includes extraction.py's plain-text FALLBACK path
    (designed to salvage a truncated completion that never got to box
    anything). That fallback matched a stray mathematical expression in
    the middle of a mid-reasoning step and reported it as a "found
    answer" -- causing search to stop after just one step, before the
    model ever reached a real boxed conclusion. Fixed by requiring an
    actual boxed span to exist first (`find_boxed_spans`), and only then
    checking it extracts unambiguously -- the fallback path is never
    reachable from here.
    """
    if not find_boxed_spans(text):
        return False
    return extract_answer(text, finish_reason="stop").status == FailureStatus.OK


def bounded_beam_search(
    problem: str,
    backend: Backend,
    prm_client: PRMClient,
    decode_cfg: DecodeConfig,
    token_budget: int,
    beam_width: int = 4,
    max_steps: int = 12,
    tokens_per_step_cap: int = 256,
) -> SearchResult:
    """One bounded, PRM-guided beam search run for one problem.

    Each round: expand every active beam by one step (stopping at the
    next `\\n\\n`), score every candidate's accumulated text with the
    PRM, keep the top `beam_width` by PRM mean_reward, charge the rest
    as discarded. Stops when `token_budget` is exhausted, `max_steps` is
    reached, or every surviving beam is finished.
    """
    step_cfg = DecodeConfig(
        temperature=decode_cfg.temperature, top_p=decode_cfg.top_p,
        max_tokens=min(tokens_per_step_cap, decode_cfg.max_tokens),
        seed=decode_cfg.seed, thinking_mode=decode_cfg.thinking_mode,
        stop_sequences=(STEP_STOP,),
    )

    beams = [Beam()]
    total_tokens = 0
    discarded_branch_tokens: list[int] = []
    prm_forwards = 0
    history: list[list[Beam]] = []

    for _step in range(max_steps):
        if total_tokens >= token_budget:
            break

        candidates: list[Beam] = []
        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue
            prompt = _build_prompt(problem, beam)
            [sample] = backend.generate([prompt], step_cfg)
            new_text = beam.text + sample.text
            new_tokens = beam.tokens_spent + sample.completion_tokens
            total_tokens += sample.completion_tokens
            candidates.append(Beam(text=new_text, tokens_spent=new_tokens, finished=_is_finished(new_text)))

        # Score every candidate's accumulated text -- an empty segmentation
        # (no double-newline yet, e.g. a very short first step) leaves
        # mean_reward at its default 0.0, never silently treated as a
        # confident score (invariant #6/#7: no score computed != a real
        # score of 0, but for RANKING purposes within one round, an
        # unscored beam sorting last is the correct, conservative behavior).
        for beam in candidates:
            steps = segment_double_newline(beam.text) if beam.text else []
            if steps:
                result = prm_client.score(problem, steps)
                prm_forwards += 1
                if result.ok:
                    beam.mean_reward = result.mean_reward

        candidates.sort(key=lambda b: b.mean_reward, reverse=True)
        survivors = candidates[:beam_width]
        discarded = candidates[beam_width:]
        discarded_branch_tokens.extend(b.tokens_spent for b in discarded)

        history.append(list(candidates))
        beams = survivors

        if all(b.finished for b in beams):
            break

    beams.sort(key=lambda b: b.mean_reward, reverse=True)
    winner = beams[0]
    charge = charge_search_action(
        kept_tokens=winner.tokens_spent,
        discarded_branch_token_counts=discarded_branch_tokens,
        prm_forwards=prm_forwards,
    )
    return SearchResult(final_text=winner.text, charge=charge, n_steps=len(history), beams_history=history)
