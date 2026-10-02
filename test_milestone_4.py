import pytest
import datetime
from milestone_4_platform import (
    InMemoryVectorStore, 
    IntelligentBugPlatform, 
    HistoricalDefect, 
    SubmittedBug
)

@pytest.fixture
def initialized_platform():
    vs = InMemoryVectorStore()
    platform = IntelligentBugPlatform(vs)
    
    # Add baseline defect
    vs.add_defect(HistoricalDefect(
        defect_id="BUG-TEST-1",
        title="Database Timeout",
        description="Database queries timing out under load.",
        affected_component="Database Layer",
        severity="HIGH",
        priority="P1",
        exception_type="ConnectionTimeoutException",
        root_cause="Connection leak in pool.",
        resolution="Closed active connections properly.",
        confirmed_fix="conn.close()",
        timestamp=datetime.datetime.utcnow()
    ))
    return platform, vs

def test_kb_validation_gate(initialized_platform):
    _, vs = initialized_platform
    invalid_defect = HistoricalDefect(
        defect_id="BUG-INVALID",
        title="Incomplete Record",
        description="",  # Missing required description
        affected_component="Auth",
        severity="LOW",
        priority="P3",
        exception_type="None",
        root_cause="",
        resolution="",
        confirmed_fix="",
        timestamp=datetime.datetime.utcnow()
    )
    success, msg = vs.add_defect(invalid_defect)
    assert not success
    assert "Validation Gate Failed" in msg

def test_duplicate_guard_trigger(initialized_platform):
    _, vs = initialized_platform
    duplicate_defect = HistoricalDefect(
        defect_id="BUG-DUP",
        title="Database Timeout",
        description="Database queries timing out under load.",
        affected_component="Database Layer",
        severity="HIGH",
        priority="P1",
        exception_type="ConnectionTimeoutException",
        root_cause="Connection leak in pool.",
        resolution="Closed active connections properly.",
        confirmed_fix="conn.close()",
        timestamp=datetime.datetime.utcnow()
    )
    success, msg = vs.add_defect(duplicate_defect)
    assert not success
    assert "Duplicate Guard Triggered" in msg

def test_insufficient_evidence_fallback(initialized_platform):
    platform, _ = initialized_platform
    noisy_bug = SubmittedBug(
        bug_id="NOISY-01",
        raw_description="Unknown frontend glitch.",
        raw_logs="No log output available."
    )
    report = platform.diagnose_bug(noisy_bug)
    assert report.root_cause.insufficient_evidence is True
    assert report.root_cause.confidence < 0.40