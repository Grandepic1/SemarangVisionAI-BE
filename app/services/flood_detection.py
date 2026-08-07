"""Flood detection on CCTV camera frames using the trained YOLO model.

Pipeline per camera:
    1. grab one frame from the HLS stream (OpenCV/FFmpeg backend, bounded timeout)
    2. run the trained flood model on the frame
    3. report whether a flood was detected, the confidence, and the boxes

The model is loaded lazily once and reused across requests (thread-safe).
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

from app.core.config import FLOOD_GRAB_TIMEOUT_MS, FLOOD_IMGSZ, FLOOD_MAX_WORKERS, FLOOD_MODEL_PATH

# Project root (this file lives in app/services/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_model = None
_model_lock = threading.Lock()
# Ultralytics models are NOT safe for concurrent predict() calls (they mutate
# internal module state), so inference is serialized. Frame grabs stay
# parallel — they are the slow part — and inference is fast (~100-300ms).
_infer_lock = threading.Lock()


def _resolve_model_path() -> str:
    """Resolve the configured model path to an absolute path if relative."""
    path = FLOOD_MODEL_PATH
    if not os.path.isabs(path):
        path = str(PROJECT_ROOT / path)
    return path


def get_model():
    """Return the shared YOLO model, loading it on first use (thread-safe)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO

                _model = YOLO(_resolve_model_path())
    return _model


def camera_key(camera: dict) -> tuple:
    """Stable identity for a camera: (cctv_id, camera_id)."""
    return (camera.get("cctv_id"), camera.get("camera_id"))


def grab_frame(stream_url: str) -> "cv2.typing.MatLike | None":
    """Grab a single frame from an HLS/RTSP stream, or None on failure.

    Open/read timeouts keep dead or slow streams from hanging the worker.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, FLOOD_GRAB_TIMEOUT_MS)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, FLOOD_GRAB_TIMEOUT_MS)
        ok, frame = cap.read()
        return frame if ok else None
    except Exception:
        return None
    finally:
        if cap is not None:
            cap.release()


def detect_flood(frame, conf_threshold: float) -> dict:
    """Run the flood model on a frame.

    Returns {"detected": bool, "count": int, "max_confidence": float,
    "boxes": [{"x1", "y1", "x2", "y2", "confidence"}, ...]}.
    """
    with _infer_lock:
        results = get_model().predict(
            frame,
            imgsz=FLOOD_IMGSZ,
            conf=conf_threshold,
            device="cpu",
            verbose=False,
        )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return {"detected": False, "count": 0, "max_confidence": 0.0, "boxes": []}

    confs = [round(float(c), 4) for c in boxes.conf]
    return {
        "detected": True,
        "count": len(confs),
        "max_confidence": max(confs),
        "boxes": [
            {
                "x1": round(float(b[0]), 2),
                "y1": round(float(b[1]), 2),
                "x2": round(float(b[2]), 2),
                "y2": round(float(b[3]), 2),
                "confidence": c,
            }
            for b, c in zip(boxes.xyxy, confs)
        ],
    }


def analyze_camera(camera: dict, conf_threshold: float) -> dict:
    """Grab one frame from the camera and detect flood.

    Never raises: failures are reported inline so one broken stream does not
    sink the whole batch. The camera's own coordinates (latitude/longitude)
    stay on the returned dict, so callers know *where* a flood was seen.
    """
    result = {**camera, "flood_checked": True}
    try:
        frame = grab_frame(camera.get("stream_url", ""))
        if frame is None:
            result.update(
                {
                    "flood_detected": False,
                    "flood_confidence": 0.0,
                    "flood_count": 0,
                    "flood_error": "frame grab failed",
                }
            )
            return result
        detection = detect_flood(frame, conf_threshold)
        result.update(
            {
                "flood_detected": detection["detected"],
                "flood_confidence": detection["max_confidence"],
                "flood_count": detection["count"],
            }
        )
        return result
    except Exception as exc:  # noqa: BLE001 - report and continue
        result.update(
            {
                "flood_detected": False,
                "flood_confidence": 0.0,
                "flood_count": 0,
                "flood_error": str(exc)[:200],
            }
        )
        return result


def analyze_cameras(
    cameras: list[dict],
    conf_threshold: float,
    max_workers: int = FLOOD_MAX_WORKERS,
) -> dict:
    """Analyze cameras in parallel, keyed by (cctv_id, camera_id).

    Duplicate cameras (same cctv_id/camera_id) are analyzed only once.
    """
    analyzed: dict = {}
    if not cameras:
        return analyzed
    unique = {camera_key(cam): cam for cam in cameras}.values()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_camera, cam, conf_threshold): camera_key(cam) for cam in unique}
        for future in as_completed(futures):
            key = futures[future]
            try:
                analyzed[key] = future.result()
            except Exception:  # noqa: BLE001 - analyze_camera already guards, but stay safe
                analyzed[key] = None
    return analyzed
