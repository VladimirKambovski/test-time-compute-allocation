"""
Day 5: PRM step-segmentation conventions, tried in the fallback order
configs/prms/qwen-math-prm-7b.yaml specifies: double_newline -> step_prefix
-> special_token. Pure functions, text in / steps out.

Per invariant #7 (closed failure taxonomy), a convention that finds no
steps in a given completion does NOT silently fall back to "treat the
whole completion as one step" -- that would be indistinguishable from a
convention that genuinely applies and finds exactly one step. It returns
[] instead; the caller is responsible for tagging that
`step_segmentation_failed`, never silently scoring it.

Empirical scan (200 completions from notes/scratch/day4_pool.jsonl,
2026-08-23, before trusting any convention): 100% contain at least one
blank-line break; 0% contain a literal "Step k:"/"Step k." prefix (this
policy's actual style is numbered-markdown, e.g. "1.  **Calculate
$r$:**", not literal "Step"); no recurring exotic delimiter token was
found for `special_token` to key off of. double_newline is the only
convention with universal coverage on this specific pool -- reported as
a real finding, not assumed in advance of running the scan.
"""
from __future__ import annotations

import re

_STEP_PREFIX_RE = re.compile(r"(?im)^\s*Step\s*\d+\s*[:.]")


def segment_double_newline(text: str) -> list[str]:
    """Split on one-or-more blank lines. Strips each step, drops empties."""
    steps = [s.strip() for s in re.split(r"\n\s*\n+", text)]
    return [s for s in steps if s]


def segment_step_prefix(text: str) -> list[str]:
    """Split on a literal "Step k:" / "Step k." line prefix. Returns []
    (never the whole text as one synthetic step) if the pattern never
    matches -- see module docstring: expected to return [] for most/all
    samples on the current pool, since this policy doesn't emit literal
    "Step k:" markers. That is the intended, honest behavior.
    """
    matches = list(_STEP_PREFIX_RE.finditer(text))
    if not matches:
        return []
    spans = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append(text[start:end].strip())
    return [s for s in spans if s]


def segment_special_token(text: str, token: str | None) -> list[str]:
    """Split on a literal model-specific step-separator token, if one is
    configured. `token=None` (no such token is currently known for this
    policy/PRM combination -- see module docstring) always returns [],
    never a guessed delimiter.
    """
    if not token:
        return []
    steps = [s.strip() for s in text.split(token)]
    return [s for s in steps if s]


SEGMENTERS = {
    "double_newline": lambda text, token=None: segment_double_newline(text),
    "step_prefix": lambda text, token=None: segment_step_prefix(text),
    "special_token": segment_special_token,
}


def segment(text: str, convention: str, special_token: str | None = None) -> list[str]:
    if convention not in SEGMENTERS:
        raise ValueError(f"Unknown segmentation convention {convention!r}. Valid: {sorted(SEGMENTERS)}")
    return SEGMENTERS[convention](text, special_token)
