"""
ai_api.py - Second Sentry
=========================
Send snapshot images to a Roboflow Workflow (YOLOv11) via the
Serverless Hosted API and return detected objects.

Works on Raspberry Pi 5 and Windows.
"""

import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# Load environment variables from .env file
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID")

API_URL = "https://serverless.roboflow.com"


def detect_objects(image_path):
    """
    Send an image to the Roboflow Workflow API and return detected objects.

    Parameters
    ----------
    image_path : str
        Path to the image file (JPEG or PNG).

    Returns
    -------
    list[dict]
        Each dict contains 'class', 'confidence', and 'bbox' keys.
        Returns an empty list if detection fails.
    """
    if not ROBOFLOW_API_KEY or not ROBOFLOW_WORKSPACE or not ROBOFLOW_WORKFLOW_ID:
        print("[AI_API] ERROR: ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, or ROBOFLOW_WORKFLOW_ID not set.")
        return []

    if not os.path.isfile(image_path):
        print(f"[AI_API] ERROR: Image not found: {image_path}")
        return []

    try:
        client = InferenceHTTPClient(
            api_url=API_URL,
            api_key=ROBOFLOW_API_KEY,
        )

        result = client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_WORKFLOW_ID,
            images={"image": image_path},
            use_cache=True,
        )

        # run_workflow returns a list of output dicts (one per image)
        predictions = []
        if isinstance(result, list) and result:
            predictions = result[0].get("predictions", [])
        elif isinstance(result, dict):
            predictions = result.get("predictions", [])

        detected = []
        for pred in predictions:
            detected.append({
                "class": pred.get("class", "unknown"),
                "confidence": round(pred.get("confidence", 0.0), 2),
                "bbox": {
                    "x": pred.get("x"),
                    "y": pred.get("y"),
                    "width": pred.get("width"),
                    "height": pred.get("height"),
                },
            })

        print(f"[AI_API] {len(detected)} object(s) detected.")
        return detected

    except Exception as e:
        print(f"[AI_API] ERROR: {e}")
        return []


if __name__ == "__main__":
    results = detect_objects("test.jpg")

    if results:
        print("\nDetected objects:")
        for obj in results:
            print(f"  {obj['class']} (confidence: {obj['confidence']})")
    else:
        print("No objects detected.")
