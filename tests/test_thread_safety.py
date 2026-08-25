"""
Real bug found live 2026-08-23 (Day 10) while adding the second
`test_controller_parity.py` case: `math_verify.parse()`/`verify()` use
`signal.alarm()` for their timeout, which only works on the main thread.
FastAPI/Starlette runs a sync `def` route (the actual `/solve` gateway
endpoint) in a worker thread by default -- meaning every real live
gateway request touching answer extraction would have crashed with
`ValueError: signal only works in main thread`, not a hypothetical.

This is the SECOND time this exact class of bug has appeared (first:
Day 6's scoring script, fixed by restructuring that one script to run
sequentially on the main thread -- not an option for a live,
request-serving gateway). `answers/thread_safety.py` is the general
fix; this file proves it actually holds under a real worker thread, not
just "the parity test happens to pass now."
"""

from __future__ import annotations

import concurrent.futures

from marginal_token.answers.equivalence import check_equivalent
from marginal_token.answers.extraction import extract_answer
from marginal_token.answers.taxonomy import FailureStatus


def test_extract_answer_works_from_a_worker_thread_not_just_main():
    """Before the Day-10 fix, this raised `ValueError: signal only works
    in main thread` -- every single time, deterministically, not
    flaky. Runs in a real worker thread (not the pytest main thread),
    exactly reproducing the gateway's actual execution context.
    """
    def _work():
        return extract_answer("The answer is \\boxed{4}.", finish_reason="stop")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        result = ex.submit(_work).result(timeout=10)

    assert result.status == FailureStatus.OK
    assert str(result.value) == "4"


def test_check_equivalent_works_from_a_worker_thread_not_just_main():
    def _work():
        extraction = extract_answer("\\boxed{4}", finish_reason="stop")
        return check_equivalent(prediction=extraction.value, gold="4")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        result = ex.submit(_work).result(timeout=10)

    assert result.equivalent is True
    assert result.status == FailureStatus.OK


def test_worker_thread_path_still_honors_a_real_timeout():
    """The off-main-thread path uses a dedicated executor +
    `Future.result(timeout=...)` instead of `signal.alarm()` -- confirm
    it still actually enforces the timeout (raises), rather than the fix
    accidentally disabling timeout protection altogether by passing
    `parsing_timeout=None` and forgetting to re-bound it.
    """
    from marginal_token.answers.taxonomy import FailureStatus as _FS  # noqa: F401  (re-import for clarity)
    from marginal_token.answers.thread_safety import safe_parse

    def _work():
        # An absurdly small timeout on ordinary input -- exercises the
        # timeout path deterministically without needing genuinely
        # pathological input to hang on.
        return safe_parse("$4$", parsing_timeout=0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        from math_verify.errors import TimeoutException

        try:
            ex.submit(_work).result(timeout=10)
            timed_out = False
        except TimeoutException:
            timed_out = True
    assert timed_out, "a 0-second timeout must actually fire, not be silently ignored by the worker-thread path"
