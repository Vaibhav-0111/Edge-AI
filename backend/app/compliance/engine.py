"""
Compliance Engine — converts PersonPPERecord checklists into compliance
percentages and COMPLIANT / VIOLATION status.

Required PPE set is configurable per site (config.py SiteConfig.required_ppe).
The engine also returns a list of missing PPE items for explainability — this
is what the dashboard shows in each alert (e.g. "missing: helmet, vest").
"""

import logging
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Set

from backend.app.compliance.association import PersonPPERecord

logger = logging.getLogger("edge_ai.compliance.engine")

# PPE slot names that the compliance engine understands
ALL_PPE_SLOTS: FrozenSet[str] = frozenset({"helmet", "vest", "gloves", "boots", "mask"})

# Default required PPE for a generic construction/factory site
DEFAULT_REQUIRED_PPE: FrozenSet[str] = frozenset({"helmet", "vest"})


@dataclass
class ComplianceResult:
    """
    Compliance assessment for one person in one frame.
    Designed to slot directly into the hazard engine's risk calculation.
    """
    track_id: Optional[int]
    person_bbox: List[float]
    # PPE present/absent/unknown per slot
    helmet: Optional[bool] = None
    vest: Optional[bool] = None
    gloves: Optional[bool] = None
    boots: Optional[bool] = None
    mask: Optional[bool] = None
    # Summary
    required_ppe: FrozenSet[str] = field(default_factory=lambda: DEFAULT_REQUIRED_PPE)
    missing_required: List[str] = field(default_factory=list)
    present_required: List[str] = field(default_factory=list)
    compliance_pct: float = 100.0       # 0–100
    status: str = "COMPLIANT"          # "COMPLIANT" | "VIOLATION"
    # Forwarded from association
    near_machinery: bool = False
    inside_restricted_zone: bool = False

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "person_bbox": [round(v, 1) for v in self.person_bbox],
            "helmet": self.helmet,
            "vest": self.vest,
            "gloves": self.gloves,
            "boots": self.boots,
            "mask": self.mask,
            "missing_required": self.missing_required,
            "present_required": self.present_required,
            "compliance_pct": round(self.compliance_pct, 1),
            "status": self.status,
            "near_machinery": self.near_machinery,
            "inside_restricted_zone": self.inside_restricted_zone,
        }


class ComplianceEngine:
    """
    Scores each PersonPPERecord against the required PPE set for the site.

    Scoring strategy:
      - Only slots in required_ppe count toward compliance %.
      - A slot with None (no model signal) is treated as UNKNOWN and does NOT
        count as a violation — we only fire alerts on confirmed absences (False).
      - compliance_pct = confirmed_present / required_count * 100
      - status = VIOLATION if any required slot is confirmed False (False ≠ None).
    """

    def __init__(self, required_ppe: Optional[Set[str]] = None):
        """
        Args:
            required_ppe: Set of PPE slot names mandatory at this site.
                          Defaults to {"helmet", "vest"}.
        """
        self.required_ppe: FrozenSet[str] = (
            frozenset(required_ppe) if required_ppe else DEFAULT_REQUIRED_PPE
        )
        # Validate slot names
        unknown = self.required_ppe - ALL_PPE_SLOTS
        if unknown:
            raise ValueError(f"Unknown PPE slots in required_ppe: {unknown}")

        logger.info("ComplianceEngine initialized. Required PPE: %s", self.required_ppe)

    def score(self, record: PersonPPERecord) -> ComplianceResult:
        """
        Evaluates one PersonPPERecord and returns a ComplianceResult.
        Only confirmed absences (False) trigger VIOLATION; None is ignored.
        """
        missing: List[str] = []
        present: List[str] = []

        for slot in self.required_ppe:
            value = getattr(record, slot, None)
            if value is True:
                present.append(slot)
            elif value is False:
                missing.append(slot)
            # None: no model signal — skip, don't penalise

        n_required = len(self.required_ppe)
        # compliance_pct is based on slots that gave a signal
        n_signalled = len(present) + len(missing)
        if n_signalled == 0:
            # No PPE signal at all — treat as uncertain, 100% (don't false-alarm)
            compliance_pct = 100.0
        else:
            compliance_pct = (len(present) / n_required) * 100.0

        status = "VIOLATION" if missing else "COMPLIANT"

        result = ComplianceResult(
            track_id=record.track_id,
            person_bbox=record.person_bbox,
            helmet=record.helmet,
            vest=record.vest,
            gloves=record.gloves,
            boots=record.boots,
            mask=record.mask,
            required_ppe=self.required_ppe,
            missing_required=sorted(missing),
            present_required=sorted(present),
            compliance_pct=compliance_pct,
            status=status,
            near_machinery=record.near_machinery,
            inside_restricted_zone=record.inside_restricted_zone,
        )

        if status == "VIOLATION":
            logger.debug(
                "Person %s VIOLATION — missing: %s, present: %s (%.0f%%)",
                record.track_id, missing, present, compliance_pct,
            )
        return result

    def score_all(self, records: List[PersonPPERecord]) -> List[ComplianceResult]:
        """Scores a list of PersonPPERecords and returns ComplianceResults for each."""
        return [self.score(r) for r in records]
