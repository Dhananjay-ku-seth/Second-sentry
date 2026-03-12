import platform
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import cv2
import psutil
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from backend.camera_manager import get_frame, initialize_camera, release_camera
from backend.database import (
    SUPABASE_KEY,
    SUPABASE_TABLE,
    SUPABASE_URL,
    REQUEST_TIMEOUT,
)
from backend.recognition import recognize_faces

_IS_RASPBERRY_PI = platform.system() == "Linux"


# ---------------------------------------------------------------------------
# Lifespan — camera init / teardown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_camera()
    yield
    release_camera()


app = FastAPI(title="Second Sentry", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# MJPEG stream generator
# ---------------------------------------------------------------------------

def generate_frames():
    """Yield MJPEG boundary frames with face recognition overlays."""
    while True:
        frame = get_frame()
        if frame is None:
            continue

        # Face recognition overlays
        try:
            results = recognize_faces(frame)
            for (x, y, w, h, name) in results:
                color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                )
        except Exception:
            pass  # Recognition unavailable — stream raw frame

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    """Dashboard page with live stream and links to API routes."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Second Sentry</title>
        <style>
            body { font-family: Arial, sans-serif; background: #111; color: #eee;
                   display: flex; flex-direction: column; align-items: center; padding: 2rem; }
            h1   { color: #00e676; }
            img  { border: 2px solid #00e676; border-radius: 6px; max-width: 100%; }
            nav  { margin-top: 1rem; display: flex; gap: 1rem; }
            a    { color: #00e676; text-decoration: none; padding: 0.4rem 1rem;
                   border: 1px solid #00e676; border-radius: 4px; }
            a:hover { background: #00e676; color: #111; }
        </style>
    </head>
    <body>
        <h1>&#128247; Second Sentry AI Camera</h1>
        <img src="/video" alt="Live feed">
        <nav>
            <a href="/events">Events</a>
            <a href="/system">System</a>
            <a href="/docs">API Docs</a>
        </nav>
    </body>
    </html>
    """)


@app.get("/video")
def video():
    """MJPEG live stream from the connected camera."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/events")
def events(limit: int = 20) -> JSONResponse:
    """
    Return the most recent security events from Supabase.

    Query param: ``limit`` (default 20, max 100).
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured (SUPABASE_URL / SUPABASE_KEY missing).",
        )

    limit = min(limit, 100)
    endpoint = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    params = {
        "order": "timestamp.desc",
        "limit": limit,
    }

    try:
        response = requests.get(
            endpoint, headers=headers, params=params, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Supabase request timed out.")
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Supabase error: {exc}")

    data: List[Dict[str, Any]] = response.json()
    return JSONResponse(content={"count": len(data), "events": data})


@app.get("/system")
def system() -> JSONResponse:
    """
    Return current CPU and RAM utilisation stats.
    Includes CPU temperature on Raspberry Pi.
    """
    mem = psutil.virtual_memory()

    stats: Dict[str, Any] = {
        "cpu_percent":  psutil.cpu_percent(interval=0.5),
        "cpu_cores":    psutil.cpu_count(logical=True),
        "ram_percent":  mem.percent,
        "ram_used_mb":  round(mem.used  / 1024 / 1024, 1),
        "ram_total_mb": round(mem.total / 1024 / 1024, 1),
        "platform":     platform.system(),
    }

    # CPU temperature — available on Raspberry Pi via psutil sensors
    if _IS_RASPBERRY_PI:
        try:
            temps = psutil.sensors_temperatures()
            cpu_temp = temps.get("cpu_thermal") or temps.get("coretemp")
            if cpu_temp:
                stats["cpu_temp_c"] = cpu_temp[0].current
        except AttributeError:
            pass  # psutil.sensors_temperatures not available on this OS

    return JSONResponse(content=stats)
