"""
The failure taxonomy is closed (see CLAUDE.md). An uncomputable metric
must record null + a valid status, never a silent 0/incorrect. An
unrecognized status must fail loudly.

TODO:
- assert each of the 14 valid statuses (see CLAUDE.md) is accepted
- assert an arbitrary/unknown status string raises rather than being
  coerced or silently ignored
- assert any problem with >20% non-"ok" samples gets flagged in the
  output rather than being silently included in aggregate stats
"""


def test_all_valid_statuses_accepted():
    raise NotImplementedError


def test_unknown_status_fails_loudly():
    raise NotImplementedError


def test_high_failure_rate_problems_are_flagged_not_dropped():
    raise NotImplementedError
