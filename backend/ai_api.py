import base64
import os
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration — set these in your environment or a .env file.
# Never hard-code secrets in source code.
# ---------------------------------------------------------------------------

# Your Roboflow private API key.
# Export in shell:  $env:ROBOFLOW_API_KEY = "your_key_here"
ROBOFLOW_API_KEY: Optional[str] = os.environ.get("ROBOFLOW_API_KEY")

# Full inference URL for your model, e.g.:
# https://detect.roboflow.com/<workspace>/<model>/<version>
# Export in shell:  $env:ROBOFLOW_MODEL_URL = "https://detect.roboflow.com/..."
ROBOFLOW_MODEL_URL: Optional[str] = os.environ.get("ROBOFLOW_MODEL_URL")

# Request timeout in seconds
REQUEST_TIMEOUT: int = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_objects(image_path: str) -> List[Dict[str, Any]]:
    """
    Send a local image to the Roboflow detection API and return predictions.

    The image is base64-encoded and submitted as the request body, which
    avoids multipart overhead and works well on Raspberry Pi.

    Parameters
    ----------
    image_path : str
        Path to the image file (JPEG or PNG) to analyse.

    Returns
    -------
    list[dict]
        List of prediction dicts from Roboflow, each containing keys such as:
        ``x``, ``y``, ``width``, ``height``, ``confidence``, ``class``.
        Returns an empty list if detection fails.

    Raises
    ------
    EnvironmentError
        If ROBOFLOW_API_KEY or ROBOFLOW_MODEL_URL are not set.
    FileNotFoundError
        If *image_path* does not exist.
    """
    _validate_config()

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"[AI_API] Image not found: {image_path}")

    # Encode image as base64 string
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    params = {"api_key": ROBOFLOW_API_KEY}

    try:
        response = requests.post(
            ROBOFLOW_MODEL_URL,
            params=params,
            data=encoded_image,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print(f"[AI_API] ERROR: Request timed out after {REQUEST_TIMEOUT}s.")
        return []
    except requests.exceptions.ConnectionError as exc:
        print(f"[AI_API] ERROR: Connection failed — {exc}")
        return []
    except requests.exceptions.HTTPError as exc:
        print(f"[AI_API] ERROR: HTTP {response.status_code} — {response.text}")
        return []

    data = response.json()
    predictions = data.get("predictions", [])

    print(f"[AI_API] {len(predictions)} object(s) detected in '{os.path.basename(image_path)}'.")
    return predictions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_config() -> None:
    """Raise EnvironmentError if required environment variables are missing."""
    missing = []
    if not ROBOFLOW_API_KEY:
        missing.append("ROBOFLOW_API_KEY")
    if not ROBOFLOW_MODEL_URL:
        missing.append("ROBOFLOW_MODEL_URL")
    if missing:
        raise EnvironmentError(
            f"[AI_API] Missing required environment variable(s): {', '.join(missing)}\n"
            "Set them before running:\n"
            "  $env:ROBOFLOW_API_KEY   = 'your_api_key'\n"
            "  $env:ROBOFLOW_MODEL_URL = 'https://detect.roboflow.com/<workspace>/<model>/<version>'"
        )


# ---------------------------------------------------------------------------
# Quick self-test  (python -m backend.ai_api  <path-to-image>)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m backend.ai_api <path-to-image>")

    test_image = sys.argv[1]
    results = detect_objects(test_image)

    if results:
        print("\nPredictions:")
        for pred in results:
            print(
                f"  class={pred.get('class')!r:20s}  "
                f"confidence={pred.get('confidence', 0):.2f}  "
                f"bbox=({pred.get('x')}, {pred.get('y')}, "
                f"{pred.get('width')}, {pred.get('height')})"
            )
    else:
        print("No predictions returned.")
