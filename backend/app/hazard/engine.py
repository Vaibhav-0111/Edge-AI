"""
Hazard / Risk Triage Engine — per ARCHITECTURE.md §2.5.

Computes a weighted risk score for each person from:
  - Missing required PPE (from ComplianceResult)
  - Presence of "machinery" or "vehicle" detections near the person
  - Person inside a defined restricted zone polygon
  - Fallen person or fire/smoke detection (CRITICAL override)

Risk score → severity mapping (per ARCHITECTURE.md):
  0         → SAFE
  1–39      → LOW
  40–69     → MEDIUM
  70–99     → HIGH
  100+ / override event → CRITICAL
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.app.compliance.engine import ComplianceResult
from backend.app.inference.engine import Detection

logger = logging.getLogger("edge_ai.hazard.engine")

# ---------------------------------------------------------------
# Risk weight table — matches ARCHITECTURE.md §2.5 exactly.
# Keep these in one place per RULES.md §1 (config over hardcoding).
# ---------------------------------------------------------------
DEFAULT_WEIGHTS: Dict[str, int] = {
    "missing_helmet":               30,
    "missing_vest":                 20,
    "missing_gloves_near_machinery": 20,
    "inside_restricted_zone":       30,
    "near_machinery":               20,
}

SEVERITY_THRESHOLDS = [
    (100, "CRITICAL"),
    (70,  "HIGH"),
    (40,  "MEDIUM"),
    (1,   "LOW"),
    (0,   "SAFE"),
]

# Classes from inference engine that count as hazard sources
MACHINERY_CLASSES = {"machinery", "vehicle"}
# Classes that trigger CRITICAL override regardless of score
CRITICAL_OVERRIDE_CLASSES = {"fallen-person", "Fire", "Smoke", "Fall-Detected"}


def score_to_severity(score: int) -> str:
    """Maps an integer risk score to a named severity level."""
    for threshold, label in SEVERITY_THRESHOLDS:
        if score >= threshold:
            return label
    return "SAFE"


def _bbox_centre(bbox: List[float]) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _point_in_polygon(px: float, py: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


@dataclass
class HazardResult:
    """Full risk assessment for one person in one frame."""
    track_id: Optional[int]
    person_bbox: List[float]
    risk_score: int = 0
    severity: str = "SAFE"
    reasons: List[str] = field(default_factory=list)
    critical_override: bool = False
    # Forwarded PPE state for WebSocket message
    helmet: Optional[bool] = None
    vest: Optional[bool] = None
    gloves: Optional[bool] = None
    missing_required: List[str] = field(default_factory=list)
    compliance_pct: float = 100.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "person_bbox": [round(v, 1) for v in self.person_bbox],
            "risk_score": self.risk_score,
            "severity": self.severity,
            "reasons": self.reasons,
            "critical_override": self.critical_override,
            "helmet": self.helmet,
            "vest": self.vest,
            "gloves": self.gloves,
            "missing_required": self.missing_required,
            "compliance_pct": round(self.compliance_pct, 1),
        }


class HazardEngine:
    """
    Computes per-person risk scores from compliance results + raw detections.

    The engine needs the full detection list to:
      - Locate machinery/vehicle boxes and measure distance to each person.
      - Detect fallen-person or fire/smoke classes for CRITICAL override.
      - Optionally test person centres against restricted zone polygons.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, int]] = None,
        machinery_proximity_px: float = 150.0,
        restricted_zones: Optional[List[List[Tuple[float, float]]]] = None,
    ):
        """
        Args:
            weights: Risk weight overrides. Defaults to ARCHITECTURE.md table.
            machinery_proximity_px: Pixel distance threshold to count as "near machinery".
            restricted_zones: List of polygon vertex lists [(x,y), ...].
                              Coordinates in the same pixel space as detection bboxes.
        """
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.machinery_proximity_px = machinery_proximity_px
        self.restricted_zones: List[List[Tuple[float, float]]] = restricted_zones or []

    # ------------------------------------------------------------------
    # Zone / proximity helpers
    # ------------------------------------------------------------------

    def _is_near_machinery(
        self, person_bbox: List[float], detections: List[Detection]
    ) -> bool:
        """True if any machinery/vehicle box centre is within proximity threshold."""
        pcx, pcy = _bbox_centre(person_bbox)
        for det in detections:
            if det.class_name in MACHINERY_CLASSES:
                mcx, mcy = _bbox_centre(det.bbox)
                if _euclidean((pcx, pcy), (mcx, mcy)) <= self.machinery_proximity_px:
                    return True
        return False

    def _is_in_restricted_zone(self, person_bbox: List[float]) -> bool:
        """True if person's foot centre (bottom-centre of bbox) is inside any zone."""
        cx = (person_bbox[0] + person_bbox[2]) / 2.0
        cy = person_bbox[3]  # bottom edge = feet level
        for zone in self.restricted_zones:
            if _point_in_polygon(cx, cy, zone):
                return True
        return False

    def _has_critical_override(self, detections: List[Detection]) -> Tuple[bool, List[str]]:
        """Checks for fallen-person / fire / smoke — triggers CRITICAL regardless of score."""
        reasons = []
        triggered = False
        for det in detections:
            if det.class_name in CRITICAL_OVERRIDE_CLASSES:
                reasons.append(det.class_name.lower().replace("-", "_").replace(" ", "_"))
                triggered = True
        return triggered, reasons

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score(
        self,
        compliance: ComplianceResult,
        detections: List[Detection],
    ) -> HazardResult:
        """
        Scores one person given their ComplianceResult and the full frame detections.
        Returns a HazardResult with risk_score, severity, and reasons list.
        """
        score = 0
        reasons: List[str] = []

        # --- PPE contributions ---
        if compliance.helmet is False:
            score += self.weights["missing_helmet"]
            reasons.append("missing_helmet")

        if compliance.vest is False:
            score += self.weights["missing_vest"]
            reasons.append("missing_vest")

        # Missing gloves near machinery (special combined condition)
        near_mach = self._is_near_machinery(compliance.person_bbox, detections)
        if near_mach:
            if compliance.gloves is False:
                score += self.weights["missing_gloves_near_machinery"]
                reasons.append("missing_gloves_near_machinery")
            score += self.weights["near_machinery"]
            reasons.append("near_machinery")

        # --- Zone contributions ---
        in_zone = self._is_in_restricted_zone(compliance.person_bbox)
        if in_zone:
            score += self.weights["inside_restricted_zone"]
            reasons.append("inside_restricted_zone")

        # --- CRITICAL override (fallen person / fire / smoke) ---
        override, override_reasons = self._has_critical_override(detections)
        if override:
            score = 100
            reasons.extend(override_reasons)

        severity = score_to_severity(score)

        return HazardResult(
            track_id=compliance.track_id,
            person_bbox=compliance.person_bbox,
            risk_score=score,
            severity=severity,
            reasons=reasons,
            critical_override=override,
            helmet=compliance.helmet,
            vest=compliance.vest,
            gloves=compliance.gloves,
            missing_required=compliance.missing_required,
            compliance_pct=compliance.compliance_pct,
        )

    def score_all(
        self,
        compliances: List[ComplianceResult],
        detections: List[Detection],
    ) -> List[HazardResult]:
        """Scores all persons in one frame."""
        return [self.score(c, detections) for c in compliances]
