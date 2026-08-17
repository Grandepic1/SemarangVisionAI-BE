"""Anomaly detection on CCTV camera frames using the trained YOLO model.

Detects four anomaly classes on the Semarang traffic cameras:

    kemacetan      — traffic jam / congestion
    pohon_tumbang  — fallen tree blocking the road
    konstruksi     — road construction / work zone
    kecelakaan     — traffic accident

Pipeline per camera:
    1. grab several frames from the HLS stream a few seconds apart (OpenCV/FFmpeg
       backend, bounded timeout)
    2. run the trained anomaly model on each frame
    3. report a class only when it persists across enough frames AND passes the
       per-class rules: kecelakaan must show motion (an event, not parked cars),
       konstruksi is dropped when traffic flows freely through the cones unless
       congestion is also detected

The model is loaded lazily once and reused across requests (thread-safe).
"""

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

from app.core.config import (
    ANOMALY_DEVICE,
    ANOMALY_EVENT_MOTION,
    ANOMALY_FLOW_MOTION,
    ANOMALY_GRAB_TIMEOUT_MS,
    ANOMALY_IMGSZ,
    ANOMALY_MAX_WORKERS,
    ANOMALY_MIN_FRACTION,
    ANOMALY_MODEL_PATH,
    ANOMALY_SAMPLE_FRAMES,
    ANOMALY_SAMPLE_INTERVAL_S,
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


def _run_inference(frame):
    """Run the model once on a frame; return (boxes, names)."""
    min_conf = min(
        (cfg["confidence"] for cfg in ANOMALY_CLASSES.values()), default=0.25
    )
    with _infer_lock:
        results = get_model().predict(
            frame,
            imgsz=ANOMALY_IMGSZ,
            conf=min_conf,
            device=_resolve_device(),
            verbose=False,
        )
    return results[0].boxes, results[0].names or {}


def _classify_boxes(boxes, model_names: dict) -> tuple[dict, dict]:
    """Map raw model boxes to per-class detections with thresholds applied.

    Returns (detected, box_map):
      detected = {type: [confidences]} — one entry per class above its threshold
      box_map   = {type: [[x1, y1, x2, y2], ...]} — pixel box regions, used for
                  the motion check between samples
    Classes are matched BY NAME (lowercased), so the model's class order does
    not matter — only its names.
    """
    detected: dict[str, list[float]] = {}
    box_map: dict[str, list[list[float]]] = {}
    if boxes is None or len(boxes) == 0:
        return detected, box_map

    for i, (cls_id, conf) in enumerate(
        zip(boxes.cls.int().tolist(), boxes.conf.tolist())
    ):
        cfg = _NAME_LOOKUP.get(str(model_names.get(int(cls_id), "")).lower())
        if cfg is None or conf < cfg["confidence"]:
            continue
        detected.setdefault(cfg["type"], []).append(float(conf))
        box_map.setdefault(cfg["type"], []).append([float(v) for v in boxes.xyxy[i]])
    return detected, box_map


def _classes_from_detected(detected: dict[str, list[float]]) -> list[dict]:
    """Convert {type: [confidences]} into the response shape (sorted by conf)."""
    return [
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


def detect_anomalies(frame) -> dict:
    """Run the anomaly model on a single frame.

    Returns {"classes": [{"type", "label", "confidence", "count"}, ...]} with
    one entry per detected anomaly class (sorted by confidence, highest first).
    Detections below a class's own threshold are dropped, so every reported
    class is already a confirmed anomaly.
    """
    boxes, names = _run_inference(frame)
    detected, _ = _classify_boxes(boxes, names)
    return {"classes": _classes_from_detected(detected)}


def _sample_frames(
    stream_url: str,
    n: int = ANOMALY_SAMPLE_FRAMES,
    interval_s: float = ANOMALY_SAMPLE_INTERVAL_S,
) -> list:
    """Grab up to n frames, interval_s apart; returns the frames that succeeded.

    A dead or slow stream yields fewer frames (never blocks the caller beyond
    the grab timeout per attempt).
    """
    frames = []
    for i in range(n):
        frame = grab_frame(stream_url)
        if frame is not None:
            frames.append(frame)
        if i < n - 1:
            time.sleep(interval_s)
    return frames


def _region_motion(frame_a, frame_b, boxes: list[list[float]]) -> float:
    """Fraction of changed pixels inside the union of box regions (0..1).

    Pixels whose grayscale value moved by more than 15 (of 255) between the
    two frames count as "changed". Static scenes (parked cars, closed roads)
    score ~0; flowing traffic or people moving score much higher.
    """
    if not boxes:
        return 0.0
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    h, w = gray_a.shape
    x1 = max(0, int(min(b[0] for b in boxes)))
    y1 = max(0, int(min(b[1] for b in boxes)))
    x2 = min(w, int(max(b[2] for b in boxes)))
    y2 = min(h, int(max(b[3] for b in boxes)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = thresh[y1:y2, x1:x2]
    return float(cv2.countNonZero(region)) / (region.shape[0] * region.shape[1])


def _confirm_classes(
    samples: list[dict],
    frames: list,
) -> list[dict]:
    """Apply persistence + motion rules to multi-frame detections.

    ``samples`` is one dict per grabbed frame: {"detected": {type: [conf]},
    "boxes": {type: [[x1, y1, x2, y2], ...]}}. Rules:

      * persistence:  a class must appear in >= ANOMALY_MIN_FRACTION of frames
      * kecelakaan:   box region must show motion >= ANOMALY_EVENT_MOTION
                      (an event) — suppresses parked cars read as accidents
      * konstruksi:   suppressed when traffic flows freely through the boxes
                      (motion >= ANOMALY_FLOW_MOTION) unless congestion is
                      also present — a work zone either jams traffic or closes
                      the road; cones with cars flowing past are lane guidance

    Falls back to per-frame reporting when fewer than 2 frames were grabbed
    (no temporal evidence available).
    """
    n = len(samples)
    if n == 0:
        return []

    detected_all: dict[str, list[list[float]]] = {}
    for sample in samples:
        for t, confs in sample["detected"].items():
            detected_all.setdefault(t, []).append(confs)

    min_frames = max(1, math.ceil(ANOMALY_MIN_FRACTION * n))
    can_check_motion = len(frames) >= 2

    confirmed = []
    for t, conf_lists in detected_all.items():
        if len(conf_lists) < min_frames:
            continue

        if can_check_motion:
            # Average box-region motion over consecutive frame pairs.
            motions = []
            for i in range(len(frames) - 1):
                boxes = (
                    samples[i]["boxes"].get(t, [])
                    + samples[i + 1]["boxes"].get(t, [])
                )
                if boxes:
                    motions.append(_region_motion(frames[i], frames[i + 1], boxes))
            avg_motion = sum(motions) / len(motions) if motions else 0.0

            if t == "kecelakaan" and avg_motion < ANOMALY_EVENT_MOTION:
                continue  # static scene -> parked cars / stopped, not an accident
            if t == "konstruksi":
                congestion = any("kemacetan" in s["detected"] for s in samples)
                if not congestion and avg_motion >= ANOMALY_FLOW_MOTION:
                    continue  # traffic flowing past cones -> lane guidance only

        all_conf = [c for confs in conf_lists for c in confs]
        confirmed.append(
            {
                "type": t,
                "label": _ANOMALY_BY_TYPE[t]["label"],
                "confidence": round(max(all_conf), 4),
                "count": max(len(confs) for confs in conf_lists),
            }
        )

    confirmed.sort(key=lambda c: c["confidence"], reverse=True)
    return confirmed


def analyze_camera(camera: dict) -> dict:
    """Sample the camera a few times and report anomalies that hold up.

    Each camera is grabbed ANOMALY_SAMPLE_FRAMES times (ANOMALY_SAMPLE_INTERVAL_S
    apart); detections must persist across ANOMALY_MIN_FRACTION of the frames and
    pass the per-class motion rules (see _confirm_classes). Never raises:
    failures are reported inline so one broken stream does not sink the whole
    batch. The camera's own coordinates stay on the returned dict.
    """
    result = {
        **camera,
        "anomaly_checked": True,
        "anomalies": [],
        "has_anomaly": False,
        "max_confidence": 0.0,
    }
    try:
        frames = _sample_frames(camera.get("stream_url", ""))
        if not frames:
            result["anomaly_error"] = "frame grab failed"
            return result

        samples = []
        for frame in frames:
            boxes, names = _run_inference(frame)
            detected, box_map = _classify_boxes(boxes, names)
            samples.append({"detected": detected, "boxes": box_map})

        classes = _confirm_classes(samples, frames)
        result["anomalies"] = classes
        result["has_anomaly"] = bool(classes)
        result["max_confidence"] = max(
            (c["confidence"] for c in classes), default=0.0
        )
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
