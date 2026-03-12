"""
motion_detector.py – Second Sentry
===================================
OpenCV-based motion detection via frame differencing.
Optimized for Raspberry Pi (low-res grayscale pipeline) but fully
runnable on Windows for development / testing.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Adjustable parameters
# ---------------------------------------------------------------------------

# Binary-threshold applied to the absolute frame difference.
# Pixel changes below this value are ignored as noise.
# Raise on noisy cameras; lower for higher sensitivity.
MOTION_THRESHOLD: int = 25

# Minimum contour area (pixels²) to count as real motion.
# Increase to ignore insects / compression artefacts; decrease to catch
# smaller movements.
MIN_CONTOUR_AREA: int = 500

# Gaussian blur kernel size – smooths frames before differencing.
# Must be an odd number.  Larger = more noise suppression but less detail.
_BLUR_KERNEL: tuple = (21, 21)

# Dilation iterations – fills small gaps between motion regions so they
# merge into a single contour.
_DILATE_ITERATIONS: int = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_motion(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    *,
    motion_threshold: int = MOTION_THRESHOLD,
    min_contour_area: int = MIN_CONTOUR_AREA,
) -> bool:
    """
    Detect motion between two consecutive BGR frames.

    Pipeline
    --------
    1. Convert both frames to grayscale.
    2. Apply Gaussian blur to suppress sensor / compression noise.
    3. Compute absolute pixel-level difference.
    4. Threshold the diff image to produce a binary motion mask.
    5. Dilate the mask to merge nearby regions.
    6. Find external contours and check whether any exceeds
       *min_contour_area*.

    Parameters
    ----------
    current_frame : np.ndarray
        The latest captured BGR frame.
    previous_frame : np.ndarray
        The immediately preceding BGR frame.
    motion_threshold : int, optional
        Pixel-difference threshold (0-255).  Defaults to module-level
        ``MOTION_THRESHOLD``.
    min_contour_area : int, optional
        Smallest contour area (px²) that counts as motion.  Defaults to
        module-level ``MIN_CONTOUR_AREA``.

    Returns
    -------
    bool
        ``True`` if motion is detected, ``False`` otherwise.
    """
    if current_frame is None or previous_frame is None:
        return False

    # 1. Grayscale conversion
    gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur (reduces high-frequency noise)
    gray_curr = cv2.GaussianBlur(gray_curr, _BLUR_KERNEL, 0)
    gray_prev = cv2.GaussianBlur(gray_prev, _BLUR_KERNEL, 0)

    # 3. Frame difference
    diff = cv2.absdiff(gray_prev, gray_curr)

    # 4. Binary threshold → motion mask
    _, mask = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)

    # 5. Dilate to close small gaps
    mask = cv2.dilate(mask, None, iterations=_DILATE_ITERATIONS)

    # 6. Contour detection
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Return True as soon as any contour exceeds the minimum area
    for contour in contours:
        if cv2.contourArea(contour) >= min_contour_area:
            return True

    return False


# ---------------------------------------------------------------------------
# Quick self-test  (python -m backend.motion_detector)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from backend.camera_manager import initialize_camera, get_frame, release_camera

    print("[MotionDetector] Running self-test — press 'q' to quit.")

    if not initialize_camera():
        raise SystemExit("Camera init failed.")

    previous_frame = get_frame()

    while True:
        current_frame = get_frame()
        if current_frame is None:
            break

        motion = detect_motion(current_frame, previous_frame)

        label = "MOTION DETECTED" if motion else "No motion"
        color = (0, 0, 255)       if motion else (0, 255, 0)

        display = current_frame.copy()
        cv2.putText(display, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        cv2.imshow("Second Sentry – Motion Detector Test",
                   cv2.resize(display, (960, 540)))

        previous_frame = current_frame

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    release_camera()
    cv2.destroyAllWindows()
    print("[MotionDetector] Self-test complete.")
