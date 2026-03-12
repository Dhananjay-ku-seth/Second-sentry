import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration — set in environment or .env file, never in source code.
# ---------------------------------------------------------------------------

# Your Supabase project URL, e.g. https://xyzxyz.supabase.co
# $env:SUPABASE_URL = "https://<your-project-ref>.supabase.co"
SUPABASE_URL: Optional[str] = os.environ.get("SUPABASE_URL")

# Supabase anon/service-role key (found in Project Settings → API)
# $env:SUPABASE_KEY = "your_supabase_key"
SUPABASE_KEY: Optional[str] = os.environ.get("SUPABASE_KEY")

# Table name in your Supabase database
SUPABASE_TABLE: str = "security_events"

# Request timeout in seconds
REQUEST_TIMEOUT: int = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert a security event record into the Supabase ``security_events`` table.

    Parameters
    ----------
    event_data : dict
        Must include the following keys:

        event_type       (str)  — e.g. "motion_detected", "unknown_face", "object_detected"
        camera_id        (str)  — identifier for the camera that triggered the event
        snapshot_url     (str)  — URL or file path of the saved snapshot
        objects_detected (list) — list of detected object class names, e.g. ["person", "bag"]

        ``timestamp`` is added automatically (UTC ISO-8601) if not provided.

    Returns
    -------
    dict with keys:
        success  (bool) — True if the row was inserted successfully.
        data     (dict) — Inserted row returned by Supabase (or empty dict on failure).
        message  (str)  — Human-readable status.
    """
    _validate_config()

    payload = _build_payload(event_data)

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        # Ask Supabase to return the inserted row
        "Prefer": "return=representation",
    }

    endpoint = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"

    print(
        f"[Database] Logging event: type={payload.get('event_type')!r} "
        f"camera={payload.get('camera_id')!r}"
    )

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        msg = f"Request timed out after {REQUEST_TIMEOUT}s."
        print(f"[Database] ERROR: {msg}")
        return _result(success=False, message=msg)
    except requests.exceptions.ConnectionError as exc:
        msg = f"Connection failed — {exc}"
        print(f"[Database] ERROR: {msg}")
        return _result(success=False, message=msg)
    except requests.exceptions.HTTPError:
        msg = f"HTTP {response.status_code} — {response.text}"
        print(f"[Database] ERROR: {msg}")
        return _result(success=False, message=msg)

    inserted = response.json()
    # Supabase returns a list when using Prefer: return=representation
    row = inserted[0] if isinstance(inserted, list) and inserted else inserted

    print(f"[Database] Event logged successfully (id={row.get('id', 'N/A')}).")
    return _result(success=True, data=row, message="Event logged successfully.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_payload(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise and validate the event dict before sending to Supabase.
    Adds a UTC timestamp if one is not supplied.
    """
    payload: Dict[str, Any] = {
        "event_type":       event_data.get("event_type", "unknown"),
        "camera_id":        event_data.get("camera_id", "unknown"),
        "timestamp":        event_data.get(
                                "timestamp",
                                datetime.now(timezone.utc).isoformat(),
                            ),
        "snapshot_url":     event_data.get("snapshot_url", ""),
        "objects_detected": event_data.get("objects_detected", []),
    }
    return payload


def _validate_config() -> None:
    """Raise EnvironmentError if required environment variables are missing."""
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")
    if missing:
        raise EnvironmentError(
            f"[Database] Missing required environment variable(s): {', '.join(missing)}\n"
            "Set them before running:\n"
            "  $env:SUPABASE_URL = 'https://<project-ref>.supabase.co'\n"
            "  $env:SUPABASE_KEY = 'your_supabase_key'"
        )


def _result(
    success: bool,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "success": success,
        "data":    data or {},
        "message": message,
    }


# ---------------------------------------------------------------------------
# Quick self-test  (python -m backend.database)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_event: Dict[str, Any] = {
        "event_type":       "motion_detected",
        "camera_id":        "cam_01",
        "snapshot_url":     "https://example.com/snapshots/test.jpg",
        "objects_detected": ["person"],
    }

    print("[Database] Running self-test with sample event …")
    result = log_event(test_event)

    print("\nResult:")
    for key, value in result.items():
        print(f"  {key:<10}: {value}")
