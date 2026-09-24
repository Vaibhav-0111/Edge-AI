"""Configuration settings for Edge AI PPE safety and hazard triage pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class RiskWeights:
    """Weight constants for risk score calculation per ARCHITECTURE.md."""
    missing_helmet: int = 30
    missing_vest: int = 20
    missing_gloves_near_machinery: int = 20
    inside_restricted_zone: int = 30
    near_machinery: int = 20
    fallen_person_override: bool = True
    fire_smoke_override: bool = True


@dataclass
class TemporalConfig:
    """Settings for temporal violation confirmation."""
    buffer_size: int = 5
    confirm_threshold: int = 4  # Minimum positive detections in window to confirm
    max_missed_frames: int = 15  # Reset tracker state if absent for N frames


@dataclass
class RestrictedZone:
    """Defines a polygonal or rectangular restricted zone."""
    zone_id: str
    name: str
    polygon: List[Tuple[float, float]]  # Normalized or pixel coordinates [(x, y), ...]
    severity: str = "HIGH"


@dataclass
class HazardZone:
    """Defines machinery or high-risk equipment zone."""
    hazard_id: str
    name: str
    polygon: List[Tuple[float, float]]
    requires_gloves: bool = True


@dataclass
class SiteConfig:
    """Site-level compliance and zone policies."""
    site_id: str = "SITE_FACTORY_01"
    camera_id: str = "CAM_01"
    required_ppe: Set[str] = field(default_factory=lambda: {"helmet", "vest"})
    optional_ppe: Set[str] = field(default_factory=lambda: {"gloves", "boots", "goggles"})
    restricted_zones: List[RestrictedZone] = field(default_factory=list)
    machinery_zones: List[HazardZone] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    model_path: str = "backend/models/yolov8n_ppe.onnx"
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    target_fps: int = 25
    demo_video_path: str = "dataset/raw/demo_sample.mp4"
    risk_weights: RiskWeights = field(default_factory=RiskWeights)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    site: SiteConfig = field(default_factory=SiteConfig)


# Default shared configuration instance
default_config = PipelineConfig()
