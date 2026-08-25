"""
Thread-safe wrappers around `math_verify.parse()`/`verify()`.

Real bug found live 2026-08-23 (Day 10), not hypothetical: both
functions enforce their timeout via `signal.alarm()`, which only works
on the interpreter's MAIN thread -- calling either from any other thread
raises `ValueError` immediately (found once already, in a scoring script
on Day 6; found again here in the actual live gateway path, since
Starlette runs a sync `def` FastAPI route in a worker thread by
default, not the main thread). Day 6's fix was to restructure that one
script to do extraction/equivalence sequentially on the main thread
before handing work to a thread pool -- a fine fix for a batch script,
but not applicable to a live request-serving gateway, which has no
control over which thread ASGI hands it.

Fix here is general, not another one-off restructure: on the main
thread, delegate straight to math_verify's own signal-based timeout
(cheap, no extra thread). Off the main thread, enforce the SAME
wall-clock timeout via a dedicated worker thread +
`Future.result(timeout=...)` instead of `signal.alarm()` -- this works
regardless of which thread calls it, so a live inference path degrades
to a real, honest timeout (mapped to the same `TimeoutException`
callers already handle) rather than crashing outright.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from math_verify import parse as _mv_parse
from math_verify import verify as _mv_verify
from math_verify.errors import TimeoutException

# Small, shared, lazily-used pool -- only actually spun up if something
# calls in from a non-main thread (the common case, e.g. offline batch
# scripts run entirely on the main thread and never touch this pool).
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="math_verify_timeout")


def _on_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def safe_parse(text: str, parsing_timeout: int | None = 5) -> Any:
    """Thread-safe equivalent of `math_verify.parse(text, parsing_timeout=...)`.
    Raises `math_verify.errors.TimeoutException` on timeout, from either
    thread context, so callers handle exactly one exception type
    regardless of which thread called them from.
    """
    if parsing_timeout is None or _on_main_thread():
        return _mv_parse(text, parsing_timeout=parsing_timeout)
    future = _executor.submit(_mv_parse, text, parsing_timeout=None)  # disable math_verify's own signal timeout
    try:
        return future.result(timeout=parsing_timeout)
    except FutureTimeoutError as exc:
        raise TimeoutException(f"safe_parse timed out after {parsing_timeout}s (off-main-thread path)") from exc


def safe_verify(gold: Any, target: Any, timeout_seconds: int | None = 5, raise_on_error: bool = True) -> bool:
    """Thread-safe equivalent of `math_verify.verify(gold, target, timeout_seconds=..., raise_on_error=...)`."""
    if timeout_seconds is None or _on_main_thread():
        return _mv_verify(gold, target, timeout_seconds=timeout_seconds, raise_on_error=raise_on_error)
    future = _executor.submit(_mv_verify, gold, target, timeout_seconds=None, raise_on_error=raise_on_error)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        raise TimeoutException(f"safe_verify timed out after {timeout_seconds}s (off-main-thread path)") from exc
