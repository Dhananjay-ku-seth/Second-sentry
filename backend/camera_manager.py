import platform
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
_IS_RASPBERRY_PI = platform.system() == "Linux"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_RESOLUTION: Tuple[int, int] = (1920, 1080)
DEFAULT_BRIGHTNESS: int = 20    # pixel offset added after capture (0–100)
DEFAULT_GAMMA: float = 1.5      # >1.0 brightens shadows; 1.0 = no change

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_cap: Optional[cv2.VideoCapture] = None
_brightness: int = DEFAULT_BRIGHTNESS
_gamma_lut: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_gamma_lut(gamma: float) -> np.ndarray:
    """
    Pre-compute an 8-bit lookup table for gamma correction.

    gamma > 1.0  →  brightens dark regions (good for dim USB cams)
    gamma < 1.0  →  darkens the image
    gamma = 1.0  →  identity (no change)
    """
    inv_gamma = 1.0 / gamma
    lut = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return lut


def _apply_corrections(
    frame: np.ndarray,
    brightness: int,
    gamma_lut: np.ndarray,
) -> np.ndarray:
    """
    Apply brightness offset then gamma correction to a BGR frame.

    brightness is added as a flat pixel offset; cv2.convertScaleAbs
    handles overflow clamping automatically.
    """
    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=float(brightness))
    frame = cv2.LUT(frame, gamma_lut)
    return frame


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialize_camera(
    camera_index: int = 0,
    resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
    brightness: int = DEFAULT_BRIGHTNESS,
    gamma: float = DEFAULT_GAMMA,
) -> bool:
    """
    Open the USB webcam and configure it.

    Parameters
    ----------
    camera_index : int
        OS camera index (0 = first webcam).
    resolution : (width, height)
        Requested capture resolution. Defaults to 1920×1080.
        The camera may fall back to its nearest supported resolution.
    brightness : int
        Flat pixel brightness offset applied after capture (0–100).
    gamma : float
        Gamma value for correction LUT (>1.0 brightens dark frames).

    Returns
    -------
    bool
        True if the camera was opened successfully, False otherwise.
    """
    global _cap, _brightness, _gamma_lut

    # Select the appropriate backend per platform
    if _IS_RASPBERRY_PI:
        # V4L2 gives direct control and lower latency on Linux / Pi
        backend = cv2.CAP_V4L2
    else:
        # DirectShow works reliably across Windows webcams
        backend = cv2.CAP_DSHOW

    _cap = cv2.VideoCapture(camera_index, backend)

    if not _cap.isOpened():
        print(f"[CameraManager] ERROR: Could not open camera at index {camera_index}.")
        _cap = None
        return False

    # Request resolution
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    _cap.set(cv2.CAP_PROP_FPS, 30)

    # On Raspberry Pi keep the internal buffer small to avoid stale frames
    if _IS_RASPBERRY_PI:
        _cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Discard warm-up frames so auto-exposure settles before use
    for _ in range(10):
        _cap.read()

    actual_w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[CameraManager] Camera {camera_index} opened "
        f"({'Pi/V4L2' if _IS_RASPBERRY_PI else 'Windows/DSHOW'}) "
        f"→ {actual_w}×{actual_h}"
    )

    _brightness = brightness
    _gamma_lut = _build_gamma_lut(gamma)
    return True


def get_frame() -> Optional[np.ndarray]:
    """
    Capture one frame from the webcam, apply corrections, and return it.

    Returns
    -------
    numpy.ndarray or None
        Processed BGR frame, or None if capture failed.
    """
    global _cap, _brightness, _gamma_lut

    if _cap is None or not _cap.isOpened():
        print("[CameraManager] ERROR: Camera is not initialized. Call initialize_camera() first.")
        return None

    ret, frame = _cap.read()

    if not ret or frame is None:
        print("[CameraManager] WARNING: Failed to read frame from camera.")
        return None

    frame = _apply_corrections(frame, _brightness, _gamma_lut)
    return frame


def release_camera() -> None:
    """
    Release the camera resource and reset module state.

    Safe to call even if the camera was never initialized.
    """
    global _cap

    if _cap is not None:
        _cap.release()
        _cap = None
        print("[CameraManager] Camera released.")


# ---------------------------------------------------------------------------
# Quick self-test  (python -m backend.camera_manager)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[CameraManager] Running self-test …")

    if not initialize_camera():
        raise SystemExit("Camera init failed — check connection and index.")

    print("[CameraManager] Press 'q' to quit.")
    while True:
        frame = get_frame()
        if frame is None:
            break

        # Downscale preview to fit typical monitor
        preview = cv2.resize(frame, (960, 540))
        cv2.imshow("Second Sentry – Camera Manager Test", preview)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    release_camera()
    cv2.destroyAllWindows()
    print("[CameraManager] Self-test complete.")
