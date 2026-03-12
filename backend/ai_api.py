import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_URL = os.getenv("ROBOFLOW_MODEL_URL")


def detect_objects(image_path):
    """
    Send image to Roboflow API and return detected objects.
    """

    if not ROBOFLOW_API_KEY or not ROBOFLOW_MODEL_URL:
        raise ValueError("Roboflow API key or model URL not configured.")

    try:
        with open(image_path, "rb") as img:

            response = requests.post(
                f"{ROBOFLOW_MODEL_URL}?api_key={ROBOFLOW_API_KEY}",
                files={"file": img},
                timeout=10
            )

        response.raise_for_status()

        data = response.json()

        return data.get("predictions", [])

    except requests.exceptions.Timeout:
        print("Roboflow request timed out")
        return []

    except requests.exceptions.RequestException as e:
        print(f"Roboflow request failed: {e}")
        return []

    except Exception as e:
        print(f"Unexpected error in AI API: {e}")
        return []
