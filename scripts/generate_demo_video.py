"""Generates a standardized 4-scene synthetic demo video for PPE & hazard testing."""

import os
from typing import Tuple
import cv2
import numpy as np


def draw_hazard_stripes(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], stripe_w: int = 15) -> None:
    """Draws diagonal yellow and black hazard warning stripes in a rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    roi = img[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if h <= 0 or w <= 0:
        return

    pattern = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(-h, w + h, stripe_w * 2):
        pts = np.array([
            [i, 0],
            [i + stripe_w, 0],
            [i + stripe_w - h, h],
            [i - h, h]
        ], dtype=np.int32)
        cv2.fillPoly(pattern, [pts], (0, 215, 255))  # Bright yellow

    # Blend with semi-transparency
    cv2.addWeighted(pattern, 0.45, roi, 0.55, 0, dst=roi)
    cv2.rectangle(img, pt1, pt2, (0, 0, 220), 2)


def draw_worker(
    frame: np.ndarray,
    center_x: int,
    center_y: int,
    has_helmet: bool = True,
    has_vest: bool = True,
    has_gloves: bool = True,
    fallen: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Renders a stylized human figure with or without PPE items.
    Returns the person bounding box [x1, y1, x2, y2].
    """
    if fallen:
        # Horizontal figure on ground
        w, h = 130, 45
        x1, y1 = center_x - w // 2, center_y - h // 2
        x2, y2 = x1 + w, y1 + h

        # Torso / legs
        cv2.rectangle(frame, (x1 + 30, y1 + 10), (x2, y2 - 5), (60, 60, 60), -1)
        if has_vest:
            cv2.rectangle(frame, (x1 + 30, y1 + 10), (x1 + 80, y2 - 5), (0, 140, 255), -1)
            cv2.line(frame, (x1 + 55, y1 + 10), (x1 + 55, y2 - 5), (220, 220, 220), 3)

        # Head
        head_center = (x1 + 20, center_y)
        cv2.circle(frame, head_center, 16, (180, 160, 140), -1)
        if has_helmet:
            cv2.ellipse(frame, (head_center[0] - 2, head_center[1] - 4), (18, 14), 0, 180, 360, (0, 220, 255), -1)

        return (x1, y1, x2, y2)

    # Upright walking figure
    w, h = 60, 150
    x1 = center_x - w // 2
    y1 = center_y - h // 2
    x2 = x1 + w
    y2 = y1 + h

    # Pants / boots
    cv2.rectangle(frame, (x1 + 10, y1 + 90), (x1 + 26, y2 - 8), (50, 50, 70), -1)
    cv2.rectangle(frame, (x1 + 34, y1 + 90), (x1 + 50, y2 - 8), (50, 50, 70), -1)
    cv2.rectangle(frame, (x1 + 8, y2 - 10), (x1 + 28, y2), (20, 20, 20), -1)  # Boots
    cv2.rectangle(frame, (x1 + 32, y2 - 10), (x1 + 52, y2), (20, 20, 20), -1)

    # Torso (Jacket / Vest)
    cv2.rectangle(frame, (x1 + 8, y1 + 38), (x2 - 8, y1 + 92), (70, 70, 70), -1)
    if has_vest:
        cv2.rectangle(frame, (x1 + 8, y1 + 38), (x2 - 8, y1 + 92), (0, 140, 255), -1)  # Hi-vis orange
        # Reflective stripes
        cv2.line(frame, (x1 + 16, y1 + 40), (x1 + 16, y1 + 90), (220, 220, 220), 3)
        cv2.line(frame, (x2 - 16, y1 + 40), (x2 - 16, y1 + 90), (220, 220, 220), 3)
        cv2.line(frame, (x1 + 8, y1 + 65), (x2 - 8, y1 + 65), (220, 220, 220), 3)

    # Arms and Gloves
    cv2.line(frame, (x1 + 8, y1 + 45), (x1 - 4, y1 + 80), (60, 60, 60), 6)
    cv2.line(frame, (x2 - 8, y1 + 45), (x2 + 4, y1 + 80), (60, 60, 60), 6)
    glove_color = (0, 165, 255) if has_gloves else (180, 160, 140)
    cv2.circle(frame, (x1 - 5, y1 + 83), 5, glove_color, -1)
    cv2.circle(frame, (x2 + 5, y1 + 83), 5, glove_color, -1)

    # Head
    head_center = (center_x, y1 + 24)
    cv2.circle(frame, head_center, 15, (180, 160, 140), -1)

    # Helmet / Hardhat (yellow)
    if has_helmet:
        cv2.ellipse(frame, (center_x, y1 + 18), (17, 13), 0, 180, 360, (0, 215, 255), -1)
        cv2.line(frame, (center_x - 20, y1 + 19), (center_x + 20, y1 + 19), (0, 200, 240), 3)  # Brim

    return (x1 - 10, y1, x2 + 10, y2)


def generate_demo_video(output_path: str = "dataset/raw/demo_sample.mp4", duration_sec: int = 15, fps: int = 25) -> str:
    """Creates a standardized 4-scene synthetic test video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 480
    total_frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter at {output_path}")

    # Restricted zone coords: [360, 180] to [620, 450]
    zone_pt1 = (360, 180)
    zone_pt2 = (620, 450)

    for i in range(total_frames):
        # Base floor and factory background
        frame = np.full((height, width, 3), (35, 38, 42), dtype=np.uint8)

        # Floor grid lines
        for y in range(200, height, 40):
            cv2.line(frame, (0, y), (width, y), (48, 52, 58), 1)

        # Safe walking lane
        cv2.rectangle(frame, (30, 200), (320, 460), (45, 50, 52), -1)
        cv2.line(frame, (30, 200), (30, 460), (0, 220, 220), 2)
        cv2.line(frame, (320, 200), (320, 460), (0, 220, 220), 2)
        cv2.putText(frame, "WALKWAY (SAFE LANE)", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)

        # Restricted Zone (Red border + warning stripes)
        draw_hazard_stripes(frame, zone_pt1, zone_pt2, stripe_w=18)
        cv2.putText(frame, "RESTRICTED ZONE - DANGER", (zone_pt1[0] + 15, zone_pt1[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Heavy Machinery equipment block
        cv2.rectangle(frame, (420, 80), (600, 175), (80, 80, 95), -1)
        cv2.rectangle(frame, (420, 80), (600, 175), (140, 140, 160), 2)
        cv2.putText(frame, "HYDRAULIC PRESS #01", (430, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "[HAZARD SOURCE]", (450, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 255), 1)

        # Scene progression
        # Scene 1: Compliant worker (frames 0 to 90) -> x: 120, y: 320
        # Scene 2: Helmet removed (frames 90 to 180) -> x moves 120 -> 260
        # Scene 3: Enters restricted zone (frames 180 to 280) -> x moves 260 -> 450
        # Scene 4: Worker falls / triggers critical alarm (frames 280 to end)

        if i < 90:
            # Scene 1: fully compliant
            scene_label = "SCENE 1: Fully Compliant Worker (Helmet + Vest + Gloves)"
            has_helmet = True
            has_vest = True
            has_gloves = True
            fallen = False
            wx = 140 + int(15 * np.sin(i * 0.1))
            wy = 330
        elif i < 180:
            # Scene 2: removes helmet
            prog = (i - 90) / 90.0
            scene_label = "SCENE 2: Helmet Removed -> Missing PPE Violation"
            has_helmet = False
            has_vest = True
            has_gloves = True
            fallen = False
            wx = int(155 + prog * 100)
            wy = 330
        elif i < 280:
            # Scene 3: enters restricted zone near machinery
            prog = (i - 180) / 100.0
            scene_label = "SCENE 3: Restricted Zone Entry + Machinery Proximity (CRITICAL)"
            has_helmet = False
            has_vest = True
            has_gloves = False
            fallen = False
            wx = int(255 + prog * 200)
            wy = int(330 - prog * 40)
        else:
            # Scene 4: fallen person
            scene_label = "SCENE 4: Worker Slip / Fall Detected (CRITICAL OVERRIDE)"
            has_helmet = False
            has_vest = True
            has_gloves = False
            fallen = True
            wx = 460
            wy = 320

        # Draw worker
        draw_worker(frame, wx, wy, has_helmet=has_helmet, has_vest=has_vest, has_gloves=has_gloves, fallen=fallen)

        # Header overlay
        cv2.rectangle(frame, (0, 0), (width, 40), (20, 20, 24), -1)
        cv2.putText(frame, f"CCTV CAM_01 | Frame {i:04d} | {scene_label}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 180), 1)

        out.write(frame)

    out.release()
    print(f"Generated standardized demo video: {output_path} ({total_frames} frames, {fps} fps)")
    return output_path


if __name__ == "__main__":
    generate_demo_video()
