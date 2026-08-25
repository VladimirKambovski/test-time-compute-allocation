"""
Step-segmentation conventions for PRM scoring, per
`configs/prms/qwen-math-prm-7b.yaml`'s documented fallback order:
`double_newline` -> `step_prefix` -> `special_token`.

Promoted from `notes/scratch/day5_segmentation.py`, unchanged in logic --
Day 5 already settled on `double_newline` for the primary policy/PRM
combination (Gate G3: AUROC=0.9934, `notes/2026-08-23.md`), and this is
the productionized version of that same code, not a rewrite.

Per invariant #7 (closed failure taxonomy), a convention that finds no
steps in a given completion does NOT silently fall back to "treat the
whole completion as one step" -- that would be indistinguishable from a
convention that genuinely applies and finds exactly one step. It returns
`[]` instead; callers map that to `FailureStatus.STEP_SEGMENTATION_FAILED`
(see `pipeline.py`), never a guessed single-step fallback.

Empirical basis for the default (`double_newline`), from Day 5's scan of
200 real completions from this policy: 100% coverage, vs. 0% for a
literal "Step k:" prefix (this policy's actual style is numbered
markdown, e.g. "1.  **Calculate $r$:**") and no known applicable
`special_token`.
"""

from __future__ import annotations

import re

SegmentationConvention = str  # "double_newline" | "step_prefix" | "special_token"

DEFAULT_CONVENTION: SegmentationConvention = "double_newline"

_STEP_PREFIX_RE = re.compile(r"(?im)^\s*Step\s*\d+\s*[:.]")


def segment_double_newline(text: str) -> list[str]:
    """Split on one-or-more blank lines. Strips each step, drops empties."""
    steps = [s.strip() for s in re.split(r"\n\s*\n+", text)]
    return [s for s in steps if s]


def segment_step_prefix(text: str) -> list[str]:
    """Split on a literal "Step k:" / "Step k." line prefix. Returns []
    (never the whole text as one synthetic step) if the pattern never
    matches.
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
    policy/PRM combination) always returns [], never a guessed delimiter.
    """
    if not token:
        return []
    steps = [s.strip() for s in text.split(token)]
    return [s for s in steps if s]


_SEGMENTERS = {
    "double_newline": lambda text, token=None: segment_double_newline(text),
    "step_prefix": lambda text, token=None: segment_step_prefix(text),
    "special_token": segment_special_token,
}


def segment(text: str, convention: SegmentationConvention = DEFAULT_CONVENTION, special_token: str | None = None) -> list[str]:
    if convention not in _SEGMENTERS:
        raise ValueError(f"Unknown segmentation convention {convention!r}. Valid: {sorted(_SEGMENTERS)}")
    return _SEGMENTERS[convention](text, special_token)
