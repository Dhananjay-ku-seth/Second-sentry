import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# Minimum contour area (in pixels) to count as real motion.
# Increase to ignore small movements (insects, noise); decrease to be more sensitive.
MOTION_THRESHOLD: int = 500

# Gaussian blur kernel size — smooths frames before differencing to reduce noise.
# Must be an odd number.
BLUR_KERNEL_SIZE: int = 21

# Binary threshold applied to the diff image.
# Pixels brighter than this (after blur) are marked as changed.
DIFF_THRESHOLD: int = 25


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_motion(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    threshold: int = MOTION_THRESHOLD,
) -> bool:
    """
    Detect motion between two consecutive BGR frames using frame differencing.

    Steps
    -----
    1. Convert both frames to grayscale.
    2. Blur to suppress camera noise.
    3. Compute absolute pixel difference.
    4. Threshold the diff to produce a binary mask.
    5. Find contours in the mask.
    6. Return True if any contour exceeds *threshold* area.

    Parameters
    ----------
    current_frame : numpy.ndarray
        The latest captured BGR frame.
    previous_frame : numpy.ndarray
        The immediately preceding BGR frame.
    threshold : int
        Minimum contour area (pixels²) to be classified as motion.
        Defaults to the module-level MOTION_THRESHOLD.

    Returns
    -------
    bool
        True if motion was detected, False otherwise.
    """
    if current_frame is None or previous_frame is None:
        return False

    # 1. Grayscale
    gray_current  = cv2.cvtColor(current_frame,  cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # 2. Blur (reduce sensor / compression noise)
    gray_current  = cv2.GaussianBlur(gray_current,  (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    gray_previous = cv2.GaussianBlur(gray_previous, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)

    # 3. Absolute frame difference
    diff = cv2.absdiff(gray_previous, gray_current)

    # 4. Binary threshold → motion mask
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Dilate to fill small gaps between motion regions
    mask = cv2.dilate(mask, None, iterations=2)

    # 5. Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Check if any contour is large enough to be real motion
    for contour in contours:
        if cv2.contourArea(contour) >= threshold:
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
