"""
PPE Association — assigns PPE detections to their nearest/containing Person box.

Strategy (two-pass):
  Pass 1 — Explicit negative classes: The Hansung-Cho model fires dedicated
            "NO-Hardhat" and "NO-Safety Vest" boxes. If such a box overlaps a
            Person box (IoU > 0 or centre inside), treat it as a confirmed
            absence for that person.
  Pass 2 — Positive PPE classes: "Hardhat" and "Safety Vest" boxes are
            spatially associated to the nearest Person whose bounding box
            vertically and horizontally contains the PPE centre.

This dual strategy is important because the model was trained this way:
  - It fires NO-Hardhat on the head region of a bare-headed worker.
  - It fires Hardhat on the head region of a helmet-wearing worker.
So we honour both signals rather than re-deriving absence from silence alone.

Output — one PersonPPERecord per detected Person:
  {
      "person_bbox": [x1, y1, x2, y2],
      "track_id":    int or None,
      "helmet":      True | False | None   (None = model gave no signal)
      "vest":        True | False | None
      "gloves":      None                  (not in this model)
      "boots":       None                  (not in this model)
      "mask":        True | False | None
      "near_machinery": bool               (populated by hazard engine)
  }
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.app.inference.engine import Detection

logger = logging.getLogger("edge_ai.compliance.association")

# Map model class names to internal PPE slot names
# Classes from Hansung-Cho model:
#   Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person,
#   Safety Cone, Safety Vest, machinery, vehicle
_POSITIVE_PPE_MAP: Dict[str, str] = {
    "Hardhat":     "helmet",
    "Safety Vest": "vest",
    "Mask":        "mask",
    "Gloves":      "gloves",   # not in current model, reserved
    "Boots":       "boots",    # not in current model, reserved
}

_NEGATIVE_PPE_MAP: Dict[str, str] = {
    "NO-Hardhat":     "helmet",
    "NO-Safety Vest": "vest",
    "NO-Mask":        "mask",
    "NO-Gloves":      "gloves",
}

_PERSON_CLASS = "Person"


@dataclass
class PersonPPERecord:
    """All PPE state associated with one detected person in a frame."""
    person_bbox: List[float]          # [x1, y1, x2, y2]
    track_id: Optional[int] = None
    person_conf: float = 0.0
    # PPE slots: True=present, False=confirmed missing, None=no signal
    helmet: Optional[bool] = None
    vest: Optional[bool] = None
    gloves: Optional[bool] = None
    boots: Optional[bool] = None
    mask: Optional[bool] = None
    # Set by hazard engine later
    near_machinery: bool = False
    inside_restricted_zone: bool = False

    def to_dict(self) -> dict:
        """Serializes to a JSON-friendly dict for downstream engines."""
        return {
            "person_bbox": [round(v, 1) for v in self.person_bbox],
            "track_id": self.track_id,
            "person_conf": round(self.person_conf, 3),
            "helmet": self.helmet,
            "vest": self.vest,
            "gloves": self.gloves,
            "boots": self.boots,
            "mask": self.mask,
            "near_machinery": self.near_machinery,
            "inside_restricted_zone": self.inside_restricted_zone,
        }


def _bbox_iou(a: List[float], b: List[float]) -> float:
    """Computes IoU of two [x1, y1, x2, y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def _centre(bbox: List[float]) -> Tuple[float, float]:
    """Returns (cx, cy) of a bounding box."""
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _centre_inside(ppe_bbox: List[float], person_bbox: List[float]) -> bool:
    """True if the centre of ppe_bbox is contained within person_bbox."""
    cx, cy = _centre(ppe_bbox)
    return (person_bbox[0] <= cx <= person_bbox[2] and
            person_bbox[1] <= cy <= person_bbox[3])


def _nearest_person_idx(
    ppe_bbox: List[float],
    persons: List[PersonPPERecord],
    iou_threshold: float = 0.0,
) -> Optional[int]:
    """
    Returns the index of the Person record best matching this PPE box.
    Priority:
      1. Person whose bbox contains the PPE centre (unambiguous).
      2. Person with highest IoU above iou_threshold.
      3. Person with smallest Euclidean distance from PPE centre to person centre.
    Returns None if no persons exist.
    """
    if not persons:
        return None

    ppe_cx, ppe_cy = _centre(ppe_bbox)

    # Priority 1: centre containment
    for i, p in enumerate(persons):
        if _centre_inside(ppe_bbox, p.person_bbox):
            return i

    # Priority 2: highest IoU
    best_iou_idx = -1
    best_iou = iou_threshold
    for i, p in enumerate(persons):
        iou = _bbox_iou(ppe_bbox, p.person_bbox)
        if iou > best_iou:
            best_iou = iou
            best_iou_idx = i
    if best_iou_idx >= 0:
        return best_iou_idx

    # Priority 3: nearest centre (fallback, assign to nearest person)
    best_dist = float("inf")
    best_dist_idx = 0
    for i, p in enumerate(persons):
        pcx, pcy = _centre(p.person_bbox)
        dist = ((ppe_cx - pcx) ** 2 + (ppe_cy - pcy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_dist_idx = i
    return best_dist_idx


class PPEAssociator:
    """
    Associates all PPE detections from one frame to Person bounding boxes,
    producing a list of PersonPPERecord objects.
    """

    def __init__(
        self,
        person_conf_min: float = 0.4,
        ppe_conf_min: float = 0.30,
        overlap_iou_min: float = 0.0,
    ):
        """
        Args:
            person_conf_min: Minimum confidence to accept a Person detection.
            ppe_conf_min:    Minimum confidence to accept a PPE detection.
            overlap_iou_min: Minimum IoU for priority-2 person matching.
        """
        self.person_conf_min = person_conf_min
        self.ppe_conf_min = ppe_conf_min
        self.overlap_iou_min = overlap_iou_min

    def associate(self, detections: List[Detection]) -> List[PersonPPERecord]:
        """
        Groups detections from one frame into per-person PPE records.

        Returns a list of PersonPPERecord — one per detected Person.
        If no Person is detected, returns an empty list.
        """
        # Separate persons from PPE detections
        persons: List[PersonPPERecord] = []
        ppe_detections: List[Detection] = []

        for det in detections:
            if det.class_name == _PERSON_CLASS and det.confidence >= self.person_conf_min:
                persons.append(PersonPPERecord(
                    person_bbox=det.bbox,
                    track_id=det.track_id,
                    person_conf=det.confidence,
                ))
            elif det.confidence >= self.ppe_conf_min:
                ppe_detections.append(det)

        if not persons:
            logger.debug("No persons detected in frame, skipping association.")
            return []

        # Associate each PPE detection to the best-matching person
        for ppe in ppe_detections:
            cls = ppe.class_name
            person_idx = _nearest_person_idx(ppe.bbox, persons, self.overlap_iou_min)
            if person_idx is None:
                continue

            record = persons[person_idx]

            if cls in _POSITIVE_PPE_MAP:
                slot = _POSITIVE_PPE_MAP[cls]
                # Only set to True if not already confirmed False (negative wins)
                current = getattr(record, slot)
                if current is not False:  # don't override a confirmed absence
                    setattr(record, slot, True)

            elif cls in _NEGATIVE_PPE_MAP:
                slot = _NEGATIVE_PPE_MAP[cls]
                # Negative class = confirmed missing; overrides any positive signal
                setattr(record, slot, False)

        # For any person with no signal at all for a slot, leave it as None
        logger.debug(
            "Associated %d PPE detections to %d person(s).",
            len(ppe_detections),
            len(persons),
        )
        return persons
