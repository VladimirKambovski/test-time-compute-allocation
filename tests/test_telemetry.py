"""
`telemetry/` was already built (Day-4 safe list) but had zero test
coverage until Day 9. No `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are
configured in this environment (checked directly, not assumed) -- which
means testing the local-JSONL fallback path here is exercising the
REAL path this project actually runs on, not a mocked stand-in for a
backend nobody's connected.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from marginal_token.telemetry.logger import DecisionRecord, TelemetryLogger, new_trace_id


def test_no_langfuse_credentials_configured_in_this_environment():
    """Documents the real environment state this test suite runs
    against, so a future reader doesn't have to guess why the fallback
    path is what's being tested here.
    """
    assert not os.environ.get("LANGFUSE_PUBLIC_KEY")
    assert not os.environ.get("LANGFUSE_SECRET_KEY")


def test_new_trace_id_is_unique_each_call():
    ids = {new_trace_id() for _ in range(100)}
    assert len(ids) == 100


def test_logger_backend_is_local_fallback_without_langfuse_credentials():
    with tempfile.TemporaryDirectory() as tmp:
        logger = TelemetryLogger(local_fallback_path=Path(tmp) / "telemetry.jsonl")
        assert logger.backend == "local_jsonl_fallback"


def test_record_writes_a_complete_decision_to_the_local_fallback_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "telemetry.jsonl"
        logger = TelemetryLogger(local_fallback_path=path)
        trace_id = new_trace_id()
        record = DecisionRecord(
            trace_id=trace_id, stage="probe_scored", action="STOP", granted_tokens=0,
            metadata={"probe_agreement": 0.9},
        )
        logger.record(record)

        assert path.exists()
        lines = [json.loads(line) for line in open(path) if line.strip()]
        assert len(lines) == 1
        assert lines[0]["trace_id"] == trace_id
        assert lines[0]["stage"] == "probe_scored"
        assert lines[0]["action"] == "STOP"
        assert lines[0]["granted_tokens"] == 0
        assert lines[0]["metadata"] == {"probe_agreement": 0.9}


def test_record_appends_never_overwrites_prior_decisions():
    """A full decision PATH is multiple records sharing one trace_id --
    the fallback file must accumulate them, not clobber the previous
    entry, or a multi-stage decision trail would be unrecoverable.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "telemetry.jsonl"
        logger = TelemetryLogger(local_fallback_path=path)
        trace_id = new_trace_id()
        logger.record(DecisionRecord(trace_id=trace_id, stage="probe_scored", action=None, granted_tokens=None))
        logger.record(DecisionRecord(trace_id=trace_id, stage="action_chosen", action="SAMPLE", granted_tokens=32))
        logger.record(DecisionRecord(trace_id=trace_id, stage="outcome", action="SAMPLE", granted_tokens=32,
                                       metadata={"final_answer": "42", "correct": True}))

        lines = [json.loads(line) for line in open(path) if line.strip()]
        assert len(lines) == 3
        assert [line["stage"] for line in lines] == ["probe_scored", "action_chosen", "outcome"]
        assert all(line["trace_id"] == trace_id for line in lines)


def test_record_never_raises_even_if_the_fallback_directory_is_freshly_created():
    """Telemetry is documented as observability, not something that
    should block the caller -- verify the parent directory is created
    on demand rather than requiring the caller to have set it up.
    """
    with tempfile.TemporaryDirectory() as tmp:
        nested_path = Path(tmp) / "does" / "not" / "exist" / "yet" / "telemetry.jsonl"
        logger = TelemetryLogger(local_fallback_path=nested_path)
        logger.record(DecisionRecord(trace_id=new_trace_id(), stage="probe_scored", action=None, granted_tokens=None))
        assert nested_path.exists()
