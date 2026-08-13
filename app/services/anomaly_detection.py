"""Anomaly detection on CCTV camera frames using the trained YOLO model.

Detects four anomaly classes on the Semarang traffic cameras:

    kemacetan      — traffic jam / congestion
    pohon_tumbang  — fallen tree blocking the road
    konstruksi     — road construction / work zone
    kecelakaan     — traffic accident

Pipeline per camera:
    1. grab one frame from the HLS stream (OpenCV/FFmpeg backend, bounded timeout)
    2. run the trained anomaly model on the frame
    3. report every detected anomaly class with its confidence and box count

The model is loaded lazily once and reused across requests (thread-safe).
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

from app.core.config import (
    ANOMALY_DEVICE,
    ANOMALY_GRAB_TIMEOUT_MS,
    ANOMALY_IMGSZ,
    ANOMALY_MAX_WORKERS,
    ANOMALY_MODEL_PATH,
)

# Project root (this file lives in app/services/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Canonical anomaly classes. Each class has its own minimum confidence
# threshold — rare, serious classes (kecelakaan) need higher evidence than
# common ones. The training data.yaml must use these exact names (see
# README); detections are matched to the model BY NAME (see detect_anomalies),
# so the class order in a custom model cannot silently corrupt results.
ANOMALY_CLASSES: dict[int, dict] = {
    0: {"type": "kemacetan", "label": "Kemacetan", "confidence": 0.35},
    1: {"type": "pohon_tumbang", "label": "Pohon Tumbang", "confidence": 0.40},
    2: {"type": "konstruksi", "label": "Konstruksi", "confidence": 0.50},
    3: {"type": "kecelakaan", "label": "Kecelakaan", "confidence": 0.60},
}

# Convenience lookups over ANOMALY_CLASSES.
_ANOMALY_BY_TYPE = {cfg["type"]: cfg for cfg in ANOMALY_CLASSES.values()}
_NAME_LOOKUP = {cfg["type"].lower(): cfg for cfg in ANOMALY_CLASSES.values()}

_model = None
_model_lock = threading.Lock()
_device = None  # cached inference device, resolved lazily (see _resolve_device)
# Ultralytics models are NOT safe for concurrent predict() calls (they mutate
# internal module state), so inference is serialized. Frame grabs stay
# parallel — they are the slow part — and inference is fast (~100-300ms).
_infer_lock = threading.Lock()


def _resolve_model_path() -> str:
    """Resolve the configured model path to an absolute path if relative."""
    path = ANOMALY_MODEL_PATH
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
                _check_model_classes(_model)
    return _model


def _resolve_device() -> str:
    """Return the inference device, resolved once and cached.

    ANOMALY_DEVICE="auto" (default) uses the first CUDA GPU when torch can
    see one and falls back to CPU otherwise; an explicit value ("cpu", "0",
    "0,1") is used as-is.
    """
    global _device
    if _device is None:
        with _model_lock:
            if _device is None:
                if ANOMALY_DEVICE == "auto":
                    try:
                        import torch

                        _device = "0" if torch.cuda.is_available() else "cpu"
                    except Exception:  # noqa: BLE001 - torch import failure -> CPU
                        _device = "cpu"
                else:
                    _device = ANOMALY_DEVICE
    return _device


def _check_model_classes(model) -> None:
    """Warn loudly if the loaded model's class names don't match the expected ones.

    The committed models/best.pt is the anomaly model; a stale flood-era model
    (single "Banjir" class) would otherwise be silently interpreted as
    kemacetan because both occupy class id 0.
    """
    known = set(_NAME_LOOKUP)
    model_names = {str(name).lower() for name in (model.names or {}).values()}
    unknown = model_names - known
    missing = known - model_names
    if unknown or missing:
        print(
            "[anomaly_detection] WARNING: model class names do not match "
            f"ANOMALY_CLASSES. Model: {sorted(model_names)} | Expected: "
            f"{sorted(known)}. Mismatches will be ignored (unknown={sorted(unknown)}, "
            f"missing={sorted(missing)})."
        )


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
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ANOMALY_GRAB_TIMEOUT_MS)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, ANOMALY_GRAB_TIMEOUT_MS)
        ok, frame = cap.read()
        return frame if ok else None
    except Exception:
        return None
    finally:
        if cap is not None:
            cap.release()


def detect_anomalies(frame) -> dict:
    """Run the anomaly model on a frame.

    Returns {"classes": [{"type", "label", "confidence", "count"}, ...]} with
    one entry per detected anomaly class (sorted by confidence, highest first).
    Detections below a class's own threshold are dropped, so every reported
    class is already a confirmed anomaly.
    """
    min_conf = min((cfg["confidence"] for cfg in ANOMALY_CLASSES.values()), default=0.25)
    with _infer_lock:
        results = get_model().predict(
            frame,
            imgsz=ANOMALY_IMGSZ,
            conf=min_conf,
            device=_resolve_device(),
            verbose=False,
        )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return {"classes": []}

    # Map detections to anomaly classes BY the model's class name (lowercased)
    # so the model's class order doesn't matter — only its names.
    model_names = results[0].names or {}
    detected: dict[str, list[float]] = {}
    for cls_id, conf in zip(boxes.cls.int().tolist(), boxes.conf.tolist()):
        cfg = _NAME_LOOKUP.get(str(model_names.get(int(cls_id), "")).lower())
        if cfg is None or conf < cfg["confidence"]:
            continue
        detected.setdefault(cfg["type"], []).append(float(conf))

    classes = [
        {
            "type": anomaly_type,
            "label": _ANOMALY_BY_TYPE[anomaly_type]["label"],
            "confidence": round(max(confs), 4),
            "count": len(confs),
        }
        for anomaly_type, confs in sorted(
            detected.items(), key=lambda kv: max(kv[1]), reverse=True
        )
    ]
    return {"classes": classes}


def analyze_camera(camera: dict) -> dict:
    """Grab one frame from the camera and detect anomalies.

    Never raises: failures are reported inline so one broken stream does not
    sink the whole batch. The camera's own coordinates (latitude/longitude)
    stay on the returned dict, so callers know *where* an anomaly was seen.
    """
    result = {
        **camera,
        "anomaly_checked": True,
        "anomalies": [],
        "has_anomaly": False,
        "max_confidence": 0.0,
    }
    try:
        frame = grab_frame(camera.get("stream_url", ""))
        if frame is None:
            result["anomaly_error"] = "frame grab failed"
            return result
        classes = detect_anomalies(frame)["classes"]
        result["anomalies"] = classes
        result["has_anomaly"] = bool(classes)
        result["max_confidence"] = max((c["confidence"] for c in classes), default=0.0)
        return result
    except Exception as exc:  # noqa: BLE001 - report and continue
        result["anomaly_error"] = str(exc)[:200]
        return result


def analyze_cameras(
    cameras: list[dict],
    max_workers: int = ANOMALY_MAX_WORKERS,
) -> dict:
    """Analyze cameras in parallel, keyed by (cctv_id, camera_id).

    Duplicate cameras (same cctv_id/camera_id) are analyzed only once.
    """
    analyzed: dict = {}
    if not cameras:
        return analyzed
    unique = {camera_key(cam): cam for cam in cameras}.values()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_camera, cam): camera_key(cam) for cam in unique}
        for future in as_completed(futures):
            key = futures[future]
            try:
                analyzed[key] = future.result()
            except Exception:  # noqa: BLE001 - analyze_camera already guards, but stay safe
                analyzed[key] = None
    return analyzed
