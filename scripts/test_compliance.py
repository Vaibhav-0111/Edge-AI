"""
Unit tests for the compliance engine and PPE associator.
Uses hand-crafted fake Detection objects — no model inference needed.
Run: .\\venv\\Scripts\\python.exe scripts/test_compliance.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.inference.engine import Detection
from backend.app.compliance.association import PPEAssociator, PersonPPERecord
from backend.app.compliance.engine import ComplianceEngine

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"

errors = []

def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}  ← {detail}")
        errors.append(label)


# ------------------------------------------------------------------
# Helper to build fake detections quickly
# ------------------------------------------------------------------
def det(cls: str, conf: float, x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(class_name=cls, confidence=conf, bbox=[x1, y1, x2, y2])


# ------------------------------------------------------------------
# Test 1: Fully compliant worker — Hardhat + Safety Vest both detected
# ------------------------------------------------------------------
print("\n=== Test 1: Fully compliant worker ===")
associator = PPEAssociator()
engine = ComplianceEngine(required_ppe={"helmet", "vest"})

dets = [
    det("Person",      0.92, 100, 50, 250, 400),
    det("Hardhat",     0.90, 130, 55, 200, 110),   # inside person bbox
    det("Safety Vest", 0.85, 110, 160, 240, 310),  # inside person bbox
]
records = associator.associate(dets)
check("1a: One person record created", len(records) == 1)
check("1b: helmet=True", records[0].helmet is True)
check("1c: vest=True", records[0].vest is True)

results = engine.score_all(records)
check("1d: Status=COMPLIANT", results[0].status == "COMPLIANT")
check("1e: compliance_pct=100", results[0].compliance_pct == 100.0)
check("1f: No missing items", results[0].missing_required == [])
print(f"    {results[0].to_dict()}")


# ------------------------------------------------------------------
# Test 2: Worker with missing helmet — model fires NO-Hardhat
# ------------------------------------------------------------------
print("\n=== Test 2: Missing helmet (NO-Hardhat class) ===")
dets = [
    det("Person",       0.88, 100, 50, 250, 400),
    det("NO-Hardhat",   0.82, 140, 55, 205, 105),  # inside person bbox
    det("Safety Vest",  0.87, 110, 160, 240, 310),
]
records = associator.associate(dets)
check("2a: One person record", len(records) == 1)
check("2b: helmet=False (NO-Hardhat signal)", records[0].helmet is False)
check("2c: vest=True", records[0].vest is True)

results = engine.score_all(records)
check("2d: Status=VIOLATION", results[0].status == "VIOLATION")
check("2e: missing=[helmet]", results[0].missing_required == ["helmet"])
check("2f: compliance_pct=50", results[0].compliance_pct == 50.0)
print(f"    {results[0].to_dict()}")


# ------------------------------------------------------------------
# Test 3: Worker with both missing — NO-Hardhat + NO-Safety Vest
# ------------------------------------------------------------------
print("\n=== Test 3: Both helmet and vest missing ===")
dets = [
    det("Person",        0.91, 100, 50, 250, 400),
    det("NO-Hardhat",    0.85, 140, 55, 205, 105),
    det("NO-Safety Vest",0.79, 110, 160, 240, 310),
]
records = associator.associate(dets)
results = engine.score_all(records)
check("3a: Status=VIOLATION", results[0].status == "VIOLATION")
check("3b: missing=[helmet, vest]", sorted(results[0].missing_required) == ["helmet", "vest"])
check("3c: compliance_pct=0", results[0].compliance_pct == 0.0)
print(f"    {results[0].to_dict()}")


# ------------------------------------------------------------------
# Test 4: Two workers — one compliant, one not
# ------------------------------------------------------------------
print("\n=== Test 4: Two workers, one compliant, one not ===")
dets = [
    # Worker A (left side): compliant
    det("Person",      0.95, 50,  50, 200, 400),
    det("Hardhat",     0.91, 90,  55, 160, 110),
    det("Safety Vest", 0.88, 60, 160, 190, 310),
    # Worker B (right side): missing helmet
    det("Person",      0.90, 400, 50, 550, 400),
    det("NO-Hardhat",  0.83, 440, 55, 510, 110),
    det("Safety Vest", 0.86, 410, 160, 540, 310),
]
records = associator.associate(dets)
check("4a: Two person records", len(records) == 2)
results = engine.score_all(records)
statuses = sorted([r.status for r in results])
check("4b: One COMPLIANT, one VIOLATION", statuses == ["COMPLIANT", "VIOLATION"])
print(f"    Worker A: {results[0].to_dict()}")
print(f"    Worker B: {results[1].to_dict()}")


# ------------------------------------------------------------------
# Test 5: PPE box outside all person bboxes (far away) → fallback to nearest
# ------------------------------------------------------------------
print("\n=== Test 5: PPE box outside person bbox → nearest-person fallback ===")
dets = [
    det("Person",  0.89, 100, 100, 250, 400),
    det("Hardhat", 0.85, 300, 10, 380, 70),   # not inside person box at all
]
records = associator.associate(dets)
check("5a: Hardhat assigned to only person (nearest fallback)", records[0].helmet is True)
print(f"    {records[0].to_dict()}")


# ------------------------------------------------------------------
# Test 6: Negative overrides positive for same slot
# ------------------------------------------------------------------
print("\n=== Test 6: NO-Hardhat should override Hardhat for same person ===")
dets = [
    det("Person",    0.90, 100, 50, 250, 400),
    det("Hardhat",   0.60, 140, 55, 210, 110),  # weak positive
    det("NO-Hardhat",0.85, 145, 58, 208, 108),  # strong negative → should win
]
records = associator.associate(dets)
check("6a: helmet=False (negative wins over positive)", records[0].helmet is False)
results = engine.score_all(records)
check("6b: Status=VIOLATION", results[0].status == "VIOLATION")


# ------------------------------------------------------------------
# Test 7: No person detected → empty records
# ------------------------------------------------------------------
print("\n=== Test 7: No person detected → empty records ===")
dets = [
    det("Hardhat",     0.91, 130, 55, 200, 110),
    det("Safety Vest", 0.88, 110, 160, 240, 310),
]
records = associator.associate(dets)
check("7a: Zero records when no Person box", len(records) == 0)


# ------------------------------------------------------------------
# Test 8: to_dict() produces clean JSON-serialisable output
# ------------------------------------------------------------------
print("\n=== Test 8: to_dict() is JSON serialisable ===")
import json
dets = [
    det("Person",    0.88, 100, 50, 250, 400),
    det("NO-Hardhat",0.84, 140, 55, 205, 105),
    det("Safety Vest",0.86, 110, 160, 240, 310),
]
records = associator.associate(dets)
results = engine.score_all(records)
try:
    serialised = json.dumps(results[0].to_dict())
    check("8a: to_dict() serialises without error", True)
    check("8b: JSON contains 'status' key", '"status"' in serialised)
except Exception as e:
    check("8a: to_dict() serialises without error", False, str(e))


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print()
if errors:
    print(f"FAILED ({len(errors)} assertion(s)):", errors)
    sys.exit(1)
else:
    print("All compliance engine tests passed!")
    sys.exit(0)
