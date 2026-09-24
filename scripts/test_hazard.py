"""
Unit tests for Hazard Triage Engine & Temporal Confirmation Buffer.
Tests weighted risk scoring, zone checking, proximity, critical overrides,
and temporal multi-frame confirmation using synthetic data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.compliance.engine import ComplianceResult
from backend.app.hazard.engine import HazardEngine, HazardResult, score_to_severity
from backend.app.hazard.temporal import TemporalConfirmationBuffer
from backend.app.inference.engine import Detection


def test_score_to_severity():
    print("[1] Testing score_to_severity mapping...")
    assert score_to_severity(0) == "SAFE"
    assert score_to_severity(25) == "LOW"
    assert score_to_severity(40) == "MEDIUM"
    assert score_to_severity(65) == "MEDIUM"
    assert score_to_severity(70) == "HIGH"
    assert score_to_severity(95) == "HIGH"
    assert score_to_severity(100) == "CRITICAL"
    assert score_to_severity(130) == "CRITICAL"
    print("    PASSED score_to_severity")


def test_fully_compliant_safe():
    print("[2] Testing fully compliant person outside hazard zones...")
    engine = HazardEngine()
    comp = ComplianceResult(
        track_id=1,
        person_bbox=[100, 100, 200, 300],
        status="COMPLIANT",
        compliance_pct=100.0,
        helmet=True,
        vest=True,
        gloves=True,
    )
    result = engine.score(comp, detections=[])
    assert result.risk_score == 0
    assert result.severity == "SAFE"
    assert len(result.reasons) == 0
    print("    PASSED fully compliant safe")


def test_missing_helmet_low():
    print("[3] Testing missing helmet (+30 -> LOW)...")
    engine = HazardEngine()
    comp = ComplianceResult(
        track_id=2,
        person_bbox=[100, 100, 200, 300],
        status="VIOLATION",
        compliance_pct=66.7,
        helmet=False,
        vest=True,
    )
    result = engine.score(comp, detections=[])
    assert result.risk_score == 30
    assert result.severity == "LOW"
    assert "missing_helmet" in result.reasons
    print("    PASSED missing helmet")


def test_missing_helmet_and_vest():
    print("[4] Testing missing helmet (+30) and vest (+20) -> 50 (MEDIUM)...")
    engine = HazardEngine()
    comp = ComplianceResult(
        track_id=3,
        person_bbox=[100, 100, 200, 300],
        status="VIOLATION",
        compliance_pct=33.3,
        helmet=False,
        vest=False,
    )
    result = engine.score(comp, detections=[])
    assert result.risk_score == 50
    assert result.severity == "MEDIUM"
    assert "missing_helmet" in result.reasons
    assert "missing_vest" in result.reasons
    print("    PASSED missing helmet + vest")


def test_machinery_proximity_and_restricted_zone():
    print("[5] Testing restricted zone (+30) + machinery proximity (+20) + missing helmet (+30) -> 80 (HIGH)...")
    zone = [(50.0, 50.0), (300.0, 50.0), (300.0, 400.0), (50.0, 400.0)]
    engine = HazardEngine(restricted_zones=[zone], machinery_proximity_px=150.0)

    machinery_det = Detection(
        class_name="machinery",
        confidence=0.9,
        bbox=[220.0, 150.0, 350.0, 300.0],
    )

    comp = ComplianceResult(
        track_id=4,
        person_bbox=[100, 100, 200, 300],  # feet at y=300, inside zone
        status="VIOLATION",
        compliance_pct=66.7,
        helmet=False,
        vest=True,
    )

    result = engine.score(comp, detections=[machinery_det])
    assert result.risk_score == 80  # 30 (helmet) + 30 (zone) + 20 (machinery)
    assert result.severity == "HIGH"
    assert "inside_restricted_zone" in result.reasons
    assert "near_machinery" in result.reasons
    print("    PASSED zone + machinery + missing helmet")


def test_critical_override():
    print("[6] Testing fallen person -> CRITICAL override...")
    engine = HazardEngine()
    comp = ComplianceResult(
        track_id=5,
        person_bbox=[100, 100, 200, 300],
        status="COMPLIANT",
        compliance_pct=100.0,
        helmet=True,
        vest=True,
    )
    fallen_det = Detection(
        class_name="fallen-person",
        confidence=0.88,
        bbox=[100.0, 200.0, 300.0, 260.0],
    )
    result = engine.score(comp, detections=[fallen_det])
    assert result.critical_override is True
    assert result.risk_score >= 100
    assert result.severity == "CRITICAL"
    assert "fallen_person" in result.reasons
    print("    PASSED critical override")


def test_temporal_confirmation():
    print("[7] Testing temporal confirmation (buffer eliminates single-frame flicker)...")
    buffer = TemporalConfirmationBuffer(window_size=5, confirm_threshold=4)

    # Frame 1: Single false-positive violation frame
    h1 = HazardResult(
        track_id=10,
        person_bbox=[100, 100, 200, 300],
        risk_score=50,
        severity="MEDIUM",
        reasons=["missing_vest"],
    )
    alerts = buffer.update([h1])
    assert len(alerts) == 0, "Single frame flicker must NOT trigger an alert"

    # Frame 2: Back to safe
    h2 = HazardResult(
        track_id=10,
        person_bbox=[100, 100, 200, 300],
        risk_score=0,
        severity="SAFE",
        reasons=[],
    )
    alerts = buffer.update([h2])
    assert len(alerts) == 0

    # Frames 3, 4, 5, 6: Sustained violation (4 consecutive frames)
    h_viol = HazardResult(
        track_id=10,
        person_bbox=[100, 100, 200, 300],
        risk_score=50,
        severity="MEDIUM",
        reasons=["missing_vest"],
    )

    alerts3 = buffer.update([h_viol])
    assert len(alerts3) == 0  # 2 of last 5 frames
    alerts4 = buffer.update([h_viol])
    assert len(alerts4) == 0  # 3 of last 5 frames
    alerts5 = buffer.update([h_viol])
    assert len(alerts5) == 1, "4th violation frame in window MUST trigger confirmed alert"
    assert alerts5[0].severity == "MEDIUM"
    assert alerts5[0].track_id == 10
    print("    PASSED temporal confirmation (flicker ignored, sustained violation confirmed)")


def test_temporal_critical_immediate():
    print("[8] Testing temporal buffer fires CRITICAL override immediately...")
    buffer = TemporalConfirmationBuffer(window_size=5, confirm_threshold=4)
    h_crit = HazardResult(
        track_id=20,
        person_bbox=[100, 100, 200, 300],
        risk_score=100,
        severity="CRITICAL",
        critical_override=True,
        reasons=["fallen_person"],
    )
    alerts = buffer.update([h_crit])
    assert len(alerts) == 1, "CRITICAL override must fire immediately without waiting"
    assert alerts[0].severity == "CRITICAL"
    print("    PASSED critical override immediate firing")


if __name__ == "__main__":
    test_score_to_severity()
    test_fully_compliant_safe()
    test_missing_helmet_low()
    test_missing_helmet_and_vest()
    test_machinery_proximity_and_restricted_zone()
    test_critical_override()
    test_temporal_confirmation()
    test_temporal_critical_immediate()
    print("\nALL HAZARD AND TEMPORAL TESTS PASSED SUCCESSFULLY! (100%)")
