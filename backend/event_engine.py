"""
event_engine.py — Second Sentry core pipeline

Pipeline per frame:
  1. Receive current + previous frame from camera_manager
  2. Run motion detection (motion_detector)
  3. If motion detected:
       a. Save snapshot locally
       b. Detect objects via Roboflow (ai_api)
       c. Upload snapshot to Google Drive (cloud_upload)
       d. Log event in Supabase (database)
       e. Trigger alert
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from backend.camera_manager import get_frame, initialize_camera, release_camera
from backend.motion_detector import detect_motion
from backend.ai_api import detect_objects
from backend.cloud_upload import upload_snapshot
from backend.database import log_event

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Local folder where snapshots are saved before upload
SNAPSHOT_DIR: str = "snapshots"

# Camera identifier included in every logged event
DEFAULT_CAMERA_ID: str = "cam_01"

# JPEG compression quality for saved snapshots (0–100)
SNAPSHOT_QUALITY: int = 90


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_frame(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    camera_id: str = DEFAULT_CAMERA_ID,
) -> Optional[Dict[str, Any]]:
    """
    Run the full Second Sentry pipeline on a pair of consecutive frames.

    Parameters
    ----------
    current_frame : numpy.ndarray
        The latest BGR frame from camera_manager.
    previous_frame : numpy.ndarray
        The immediately preceding BGR frame.
    camera_id : str
        Logical identifier for the source camera.

    Returns
    -------
    dict or None
        Event result dict if motion was detected, otherwise None.
        Keys: ``motion``, ``snapshot_path``, ``objects``, ``upload``, ``db``, ``alert``.
    """
    # ------------------------------------------------------------------ 1
    # Motion detection
    # ------------------------------------------------------------------ 
    if not detect_motion(current_frame, previous_frame):
        return None  # Nothing to do

    print(f"[EventEngine] Motion detected on {camera_id}. Starting pipeline …")

    # ------------------------------------------------------------------ 2
    # Save snapshot locally
    # ------------------------------------------------------------------
    snapshot_path = _save_snapshot(current_frame)

    # ------------------------------------------------------------------ 3
    # Roboflow object detection
    # ------------------------------------------------------------------
    objects: List[Dict[str, Any]] = []
    if snapshot_path:
        try:
            objects = detect_objects(snapshot_path)
        except EnvironmentError as exc:
            print(f"[EventEngine] WARNING: Skipping AI detection — {exc}")

    object_classes = [obj.get("class", "unknown") for obj in objects]
    print(f"[EventEngine] Objects detected: {object_classes or ['none']}")

    # ------------------------------------------------------------------ 4
    # Upload snapshot to Google Drive
    # ------------------------------------------------------------------
    upload_result: Dict[str, Any] = {"success": False, "message": "No snapshot to upload."}
    if snapshot_path:
        try:
            upload_result = upload_snapshot(snapshot_path)
        except Exception as exc:
            print(f"[EventEngine] WARNING: Upload failed — {exc}")
            upload_result = {"success": False, "message": str(exc)}

    snapshot_url = snapshot_path if upload_result.get("success") else snapshot_path

    # ------------------------------------------------------------------ 5
    # Log event to Supabase
    # ------------------------------------------------------------------
    db_result: Dict[str, Any] = {"success": False, "message": "DB logging skipped."}
    event_payload = {
        "event_type":       _classify_event(object_classes),
        "camera_id":        camera_id,
        "snapshot_url":     snapshot_url or "",
        "objects_detected": object_classes,
    }
    try:
        db_result = log_event(event_payload)
    except EnvironmentError as exc:
        print(f"[EventEngine] WARNING: Skipping DB logging — {exc}")

    # ------------------------------------------------------------------ 6
    # Trigger alert
    # ------------------------------------------------------------------
    alert_result = _trigger_alert(event_payload, objects)

    # ------------------------------------------------------------------ 
    # Assemble and return result
    # ------------------------------------------------------------------
    result: Dict[str, Any] = {
        "motion":        True,
        "snapshot_path": snapshot_path,
        "objects":       objects,
        "upload":        upload_result,
        "db":            db_result,
        "alert":         alert_result,
    }

    print(
        f"[EventEngine] Pipeline complete — "
        f"snapshot={'saved' if snapshot_path else 'failed'}, "
        f"upload={'ok' if upload_result.get('success') else 'failed'}, "
        f"db={'ok' if db_result.get('success') else 'failed'}."
    )

    return result


def run_loop(camera_id: str = DEFAULT_CAMERA_ID) -> None:
    """
    Convenience function: initialise camera and run the pipeline loop.

    Blocks indefinitely. Press Ctrl+C to stop.
    """
    if not initialize_camera():
        raise RuntimeError("[EventEngine] Camera initialisation failed.")

    print("[EventEngine] Starting event loop. Press Ctrl+C to stop.")

    previous_frame = get_frame()

    try:
        while True:
            current_frame = get_frame()
            if current_frame is None:
                print("[EventEngine] WARNING: Null frame received — skipping.")
                continue

            process_frame(current_frame, previous_frame, camera_id=camera_id)
            previous_frame = current_frame

    except KeyboardInterrupt:
        print("\n[EventEngine] Stopped by user.")
    finally:
        release_camera()
        print("[EventEngine] Camera released.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_snapshot(frame: np.ndarray) -> Optional[str]:
    """
    Save a BGR frame as a timestamped JPEG in SNAPSHOT_DIR.

    Returns the file path on success, or None on failure.
    """
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    filename = f"snapshot_{timestamp}.jpg"
    file_path = os.path.join(SNAPSHOT_DIR, filename)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_QUALITY]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)

    if not success:
        print("[EventEngine] ERROR: Failed to encode snapshot.")
        return None

    with open(file_path, "wb") as f:
        f.write(buffer.tobytes())

    print(f"[EventEngine] Snapshot saved: {file_path}")
    return file_path


def _classify_event(object_classes: List[str]) -> str:
    """
    Derive a simple event type label from the detected object classes.

    Priority: person → unknown_person → object_detected → motion_detected
    """
    if not object_classes:
        return "motion_detected"
    if "person" in object_classes:
        return "person_detected"
    return "object_detected"


def _trigger_alert(
    event: Dict[str, Any],
    objects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Alert hook — extend this function to send SMS, email, push notification, etc.

    Currently prints a console alert and returns a status dict.
    """
    event_type = event.get("event_type", "unknown")
    camera_id  = event.get("camera_id",  "unknown")
    classes    = event.get("objects_detected", [])

    message = (
        f"ALERT [{event_type.upper()}] "
        f"Camera: {camera_id} | "
        f"Objects: {', '.join(classes) if classes else 'none'}"
    )

    print(f"\n{'='*60}")
    print(f"[EventEngine] {message}")
    print(f"{'='*60}\n")

    # TODO: integrate with notification service (Telegram, email, etc.)
    return {"triggered": True, "message": message}


# ---------------------------------------------------------------------------
# Entry point  (python -m backend.event_engine)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_loop()
