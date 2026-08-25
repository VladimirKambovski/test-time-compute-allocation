"""
`featurize()`: the real feature computation from docs/brief.md §16,
operating ONLY on the k=4 probe (`Probe.samples`) -- decide() must never
condition on more than the probe, since anything else would leak the
budget's own future allocation into the decision that's supposed to
choose it.

Backend-independent groups (agreement, shape, hygiene) are always
computed. The confidence group requires per-token logprobs -- available
in principle (the hosted endpoint genuinely supports it, verified
2026-08-20), but P1's entire pool (Day 6-9) was generated before
`backends/hosted_endpoint.py`'s Day-10 fix, so it has none. Per §16's
own documented design, this degrades the feature set gracefully rather
than invalidating anything: confidence features are `nan`, never a
silently-wrong 0 (0 would read as "maximally confident," the opposite
of "unknown").
"""

from __future__ import annotations

import math
from collections import Counter

from marginal_token.answers.extraction import extract_answer
from marginal_token.answers.taxonomy import FailureStatus
from marginal_token.controller.base import Probe
from marginal_token.scoring.segmentation import segment_double_newline

# Fixed, stable column order -- callers that need a numeric feature
# vector (e.g. the predictor) should always iterate in this order rather
# than relying on dict insertion order.
FEATURE_NAMES = (
    "top1_vote_fraction",
    "top2_margin",
    "normalized_entropy",
    "distinct_answer_count",
    "mean_output_length",
    "var_output_length",
    "mean_step_count",
    "extraction_failure_fraction",
    "truncation_fraction",
    "mean_logprob",
    "min_logprob",
    "self_certainty",
    "cumulative_logprob_spread",
)


def featurize(probe: Probe) -> dict[str, float]:
    samples = probe.samples
    n = len(samples)
    if n == 0:
        raise ValueError("featurize() needs at least one probe sample")

    keys: list[str | None] = []
    ok_flags: list[bool] = []
    truncated_flags: list[bool] = []
    lengths: list[int] = []
    step_counts: list[int] = []
    logprob_means: list[float] = []
    logprob_mins: list[float] = []
    self_certainties: list[float] = []
    cumulative_logprobs: list[float] = []

    for s in samples:
        extraction = extract_answer(s.text, finish_reason=s.finish_reason)
        ok = extraction.status == FailureStatus.OK
        ok_flags.append(ok)
        keys.append(str(extraction.value) if ok else None)
        truncated_flags.append(extraction.status == FailureStatus.LENGTH_TRUNCATED)
        lengths.append(s.completion_tokens)
        step_counts.append(len(segment_double_newline(s.text)))

        if s.logprobs:
            lps = [tok["logprob"] for tok in s.logprobs]
            logprob_means.append(sum(lps) / len(lps))
            logprob_mins.append(min(lps))
            cumulative_logprobs.append(sum(lps))
            certainties = []
            for tok in s.logprobs:
                top = tok.get("top_logprobs") or []
                if top:
                    mean_top = sum(t["logprob"] for t in top) / len(top)
                    certainties.append(tok["logprob"] - mean_top)
            if certainties:
                self_certainties.append(sum(certainties) / len(certainties))

    usable_keys = [k for k in keys if k is not None]
    vote_counts = Counter(usable_keys)

    if vote_counts:
        counts_sorted = sorted(vote_counts.values(), reverse=True)
        top1 = counts_sorted[0]
        top2 = counts_sorted[1] if len(counts_sorted) > 1 else 0
        # Fractions are of ALL n probe samples, not just the usable
        # subset -- so a high extraction-failure rate shows up as a low
        # top1_vote_fraction too, not hidden by a smaller denominator.
        top1_vote_fraction = top1 / n
        top2_margin = (top1 - top2) / n
        total_usable = len(usable_keys)
        probs = [c / total_usable for c in vote_counts.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(len(vote_counts)) if len(vote_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        distinct_answer_count = float(len(vote_counts))
    else:
        top1_vote_fraction = 0.0
        top2_margin = 0.0
        normalized_entropy = 0.0
        distinct_answer_count = 0.0

    mean_length = sum(lengths) / n
    var_length = sum((length - mean_length) ** 2 for length in lengths) / n

    features = {
        "top1_vote_fraction": top1_vote_fraction,
        "top2_margin": top2_margin,
        "normalized_entropy": normalized_entropy,
        "distinct_answer_count": distinct_answer_count,
        "mean_output_length": mean_length,
        "var_output_length": var_length,
        "mean_step_count": sum(step_counts) / n,
        "extraction_failure_fraction": sum(1 for ok in ok_flags if not ok) / n,
        "truncation_fraction": sum(truncated_flags) / n,
        "mean_logprob": (sum(logprob_means) / len(logprob_means)) if logprob_means else float("nan"),
        "min_logprob": min(logprob_mins) if logprob_mins else float("nan"),
        "self_certainty": (sum(self_certainties) / len(self_certainties)) if self_certainties else float("nan"),
        "cumulative_logprob_spread": (
            (max(cumulative_logprobs) - min(cumulative_logprobs)) if len(cumulative_logprobs) >= 2 else float("nan")
        ),
    }
    assert set(features) == set(FEATURE_NAMES), "featurize() output must match the declared FEATURE_NAMES exactly"
    return features
