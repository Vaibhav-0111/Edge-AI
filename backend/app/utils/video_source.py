"""Video ingestion abstraction for MP4 files, RTSP streams, and webcams."""

import logging
import time
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger("edge_ai.video_source")


class VideoSource:
    """Wraps OpenCV VideoCapture with error recovery and looping support."""

    def __init__(
        self,
        source: Union[str, int],
        loop: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        max_fps: Optional[int] = None,
    ):
        """
        Initialize video capture source.

        Args:
            source: Path to MP4, camera index (0, 1), or RTSP URL.
            loop: Whether to loop video file upon reaching EOF.
            target_size: Optional (width, height) to resize frames.
            max_fps: Optional frame rate throttle for processing.
        """
        self.source = source
        self.loop = loop
        self.target_size = target_size
        self.max_fps = max_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_file = isinstance(source, str) and not source.startswith("rtsp://")
        self._open()

    def _open(self) -> bool:
        """Opens or reopens the video capture device or file."""
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error("Failed to open video source: %s", self.source)
            return False

        logger.info("Successfully opened video source: %s", self.source)
        return True

    @property
    def fps(self) -> float:
        """Original video source frames per second."""
        if self.cap and self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30.0
        return 30.0

    @property
    def frame_count(self) -> int:
        """Total frame count if source is a file."""
        if self.cap and self.cap.isOpened() and self._is_file:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1

    @property
    def resolution(self) -> Tuple[int, int]:
        """(width, height) of native capture stream."""
        if self.cap and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (0, 0)

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator yielding (frame_index, frame_bgr).
        Degrades gracefully on dropped frames and loops files if requested.
        """
        frame_idx = 0
        interval = (1.0 / self.max_fps) if self.max_fps and self.max_fps > 0 else 0.0

        while True:
            t0 = time.time()
            if self.cap is None or not self.cap.isOpened():
                if not self._open():
                    time.sleep(1.0)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self._is_file and self.loop:
                    logger.info("End of video reached, rewinding to beginning.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        logger.warning("Could not read frame after loop rewind.")
                        break
                else:
                    logger.warning("Frame read failed or end of stream reached.")
                    break

            if self.target_size is not None:
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

            yield frame_idx, frame
            frame_idx += 1

            if interval > 0:
                elapsed = time.time() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def release(self) -> None:
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Video source released.")


def letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize and pad image while meeting stride-multiple constraints.
    Returns: (padded_image, scale_ratio, (pad_left, pad_top))
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)
