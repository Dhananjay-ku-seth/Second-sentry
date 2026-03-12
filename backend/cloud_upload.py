import os
import subprocess
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# rclone remote name as configured via `rclone config`
RCLONE_REMOTE: str = "gdrive"

# Destination folder inside Google Drive
RCLONE_DEST_FOLDER: str = "secondsentry_snapshots"

# Full destination path passed to rclone
RCLONE_DEST: str = f"{RCLONE_REMOTE}:{RCLONE_DEST_FOLDER}"

# Timeout for the rclone subprocess (seconds)
UPLOAD_TIMEOUT: int = 60


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_snapshot(file_path: str) -> Dict[str, Any]:
    """
    Upload a single image snapshot to Google Drive using rclone.

    Runs:  rclone copy <file_path> gdrive:secondsentry_snapshots

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the image file to upload.

    Returns
    -------
    dict with keys:
        success  (bool)   — True if rclone exited with code 0.
        file     (str)    — The file path that was uploaded.
        remote   (str)    — The rclone destination used.
        message  (str)    — Human-readable status message.
        stderr   (str)    — rclone stderr output (empty on success).
    """
    if not os.path.isfile(file_path):
        return _result(
            success=False,
            file=file_path,
            message=f"File not found: {file_path}",
        )

    command = ["rclone", "copy", file_path, RCLONE_DEST]

    print(f"[CloudUpload] Uploading '{os.path.basename(file_path)}' → {RCLONE_DEST} …")

    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=UPLOAD_TIMEOUT,
        )
    except FileNotFoundError:
        return _result(
            success=False,
            file=file_path,
            message="rclone is not installed or not found in PATH.",
        )
    except subprocess.TimeoutExpired:
        return _result(
            success=False,
            file=file_path,
            message=f"Upload timed out after {UPLOAD_TIMEOUT}s.",
        )

    if proc.returncode == 0:
        msg = f"Upload successful: {os.path.basename(file_path)} → {RCLONE_DEST}"
        print(f"[CloudUpload] {msg}")
        return _result(success=True, file=file_path, message=msg)
    else:
        msg = f"Upload failed (rclone exit code {proc.returncode})."
        print(f"[CloudUpload] ERROR: {msg}")
        print(f"[CloudUpload] rclone stderr: {proc.stderr.strip()}")
        return _result(
            success=False,
            file=file_path,
            message=msg,
            stderr=proc.stderr.strip(),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _result(
    success: bool,
    file: str,
    message: str,
    stderr: str = "",
) -> Dict[str, Any]:
    return {
        "success": success,
        "file": file,
        "remote": RCLONE_DEST,
        "message": message,
        "stderr": stderr,
    }


# ---------------------------------------------------------------------------
# Quick self-test  (python -m backend.cloud_upload  <path-to-image>)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m backend.cloud_upload <path-to-image>")

    status = upload_snapshot(sys.argv[1])
    print("\nUpload result:")
    for key, value in status.items():
        print(f"  {key:<10}: {value}")
