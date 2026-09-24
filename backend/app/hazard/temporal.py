"""
Temporal Confirmation Engine — per ARCHITECTURE.md §2.6.

Maintains a short rolling history buffer per tracked person to eliminate
single-frame flickering false positives (e.g. YOLO misses a helmet for 1 frame
due to occlusion or head turn).

An alert is only "confirmed" and fired once a violation appears in at least
`confirm_threshold` out of the last `window_size` frames (e.g. 4 of 5 frames).
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.app.hazard.engine import HazardResult


@dataclass
class PersonHistory:
    """Rolling state history for a single tracked person."""
    track_id: Optional[int]
    last_bbox: List[float]
    window: deque = field(default_factory=deque)  # stores list of (severity, risk_score, reasons, is_violation)
    confirmed_severity: str = "SAFE"
    last_alert_time: float = 0.0
    is_alert_active: bool = False
    last_seen_frame: int = 0


@dataclass
class ConfirmedAlert:
    """An alert that has passed temporal confirmation and is ready for broadcast."""
    track_id: Optional[int]
    person_bbox: List[float]
    risk_score: int
    severity: str
    reasons: List[str]
    compliance_pct: float
    helmet: Optional[bool]
    vest: Optional[bool]
    gloves: Optional[bool]
    confirmed_ratio: float  # e.g. 4/5 = 0.8
    is_new_alert: bool = True  # True if state just elevated or changed
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "person_bbox": [round(v, 1) for v in self.person_bbox],
            "risk_score": self.risk_score,
            "severity": self.severity,
            "reasons": self.reasons,
            "compliance_pct": round(self.compliance_pct, 1),
            "helmet": self.helmet,
            "vest": self.vest,
            "gloves": self.gloves,
            "confirmed_ratio": round(self.confirmed_ratio, 2),
            "is_new_alert": self.is_new_alert,
            "timestamp": self.timestamp,
        }


class TemporalConfirmationBuffer:
    """
    Rolling confirmation buffer across consecutive frames.
    Filters out single-frame false positives.
    """

    def __init__(
        self,
        window_size: int = 5,
        confirm_threshold: int = 4,
        alert_cooldown_seconds: float = 3.0,
        max_missing_frames: int = 30,
    ):
        """
        Args:
            window_size: Number of recent frames to consider (default: 5).
            confirm_threshold: Number of violation frames required to confirm (default: 4).
            alert_cooldown_seconds: Min seconds between duplicate alerts for same person.
            max_missing_frames: Drop history if person unseen for this many frames.
        """
        self.window_size = window_size
        self.confirm_threshold = confirm_threshold
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.max_missing_frames = max_missing_frames
        self.current_frame = 0
        self.histories: Dict[int, PersonHistory] = {}
        self._next_fallback_id = 1000

    def _iou(self, boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
        denom = boxAArea + boxBArea - interArea
        return interArea / denom if denom > 0 else 0.0

    def _resolve_track_id(self, result: HazardResult) -> int:
        if result.track_id is not None:
            return result.track_id
        
        # Spatial IoU fallback matching if track_id is None
        best_id = None
        best_iou = 0.3
        for tid, hist in self.histories.items():
            if self.current_frame - hist.last_seen_frame <= 3:
                iou = self._iou(result.person_bbox, hist.last_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
        
        if best_id is not None:
            return best_id
        
        self._next_fallback_id += 1
        return self._next_fallback_id

    def update(
        self,
        hazard_results: List[HazardResult],
        timestamp_str: Optional[str] = None,
    ) -> List[ConfirmedAlert]:
        """
        Feed in HazardResults from one frame, returns list of ConfirmedAlerts.
        """
        self.current_frame += 1
        now = time.time()
        ts = timestamp_str or time.strftime("%H:%M:%S")

        confirmed_alerts: List[ConfirmedAlert] = []
        seen_tids = set()

        for res in hazard_results:
            tid = self._resolve_track_id(res)
            seen_tids.add(tid)

            if tid not in self.histories:
                self.histories[tid] = PersonHistory(
                    track_id=res.track_id or tid,
                    last_bbox=res.person_bbox,
                    window=deque(maxlen=self.window_size),
                    last_seen_frame=self.current_frame,
                )

            hist = self.histories[tid]
            hist.last_bbox = res.person_bbox
            hist.last_seen_frame = self.current_frame

            # CRITICAL override passes immediately without waiting for buffer
            is_critical = res.critical_override or res.severity == "CRITICAL"
            is_violation = res.severity in ("MEDIUM", "HIGH", "CRITICAL")

            hist.window.append((res.severity, res.risk_score, res.reasons, is_violation))

            # Count violation frames in rolling window
            violation_count = sum(1 for _, _, _, v in hist.window if v)
            ratio = violation_count / len(hist.window) if hist.window else 0.0

            # Determine if violation is temporally confirmed
            is_confirmed = (violation_count >= self.confirm_threshold) or is_critical

            if is_confirmed and is_violation:
                # Severity is dominant severity in the window
                dominant_severity = res.severity
                is_new = (hist.confirmed_severity != dominant_severity) or (
                    now - hist.last_alert_time > self.alert_cooldown_seconds
                )

                if is_new:
                    hist.confirmed_severity = dominant_severity
                    hist.last_alert_time = now
                    hist.is_alert_active = True

                    confirmed_alerts.append(
                        ConfirmedAlert(
                            track_id=res.track_id or tid,
                            person_bbox=res.person_bbox,
                            risk_score=res.risk_score,
                            severity=dominant_severity,
                            reasons=res.reasons,
                            compliance_pct=res.compliance_pct,
                            helmet=res.helmet,
                            vest=res.vest,
                            gloves=res.gloves,
                            confirmed_ratio=ratio,
                            is_new_alert=True,
                            timestamp=ts,
                        )
                    )
            elif not is_violation:
                # Reset confirmed violation state if person is now compliant
                if hist.is_alert_active and violation_count == 0:
                    hist.confirmed_severity = "SAFE"
                    hist.is_alert_active = False

        # Clean up stale track histories
        stale_tids = [
            tid
            for tid, hist in self.histories.items()
            if self.current_frame - hist.last_seen_frame > self.max_missing_frames
        ]
        for tid in stale_tids:
            del self.histories[tid]

        return confirmed_alerts
