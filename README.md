### Semarang Vision AI (Backend)

FastAPI backend for the Semarang Vision CCTV project. Detects **road
anomalies** on live traffic cameras — kemacetan (traffic jam), pohon tumbang
(fallen tree), konstruksi (construction) and kecelakaan (accident) — and uses
them to rank navigation routes.

---

## Project structure

```
app/          FastAPI app (api/, core/, models/, services/, utils/)
models/       Production weights — models/best.pt is committed (git-tracked)
data/         Runtime data — cctvs.json (tracked, seeded into the Docker image)
scripts/      CLI tools (e.g. train_yolo.py)
datasets/     Training datasets (gitignored)
runs/         Ultralytics training output (gitignored)
notebooks/    Colab training notebook
```

---

## Model Training (YOLO11)

Train a YOLO11 detector (`yolo11n` / `yolo11s`) on your CCTV anomaly dataset.
Runs on **CPU by default** — no GPU required (the CPU-only torch build is
installed).

### Dataset format (YOLO)

```text
datasets/cctv/
├── data.yaml            # class names MUST match the order below
├── train/
│   ├── images/          # *.jpg / *.png
│   └── labels/          # *.txt  (one line per object: class x y w h, normalized)
└── val/
    ├── images/
    └── labels/
```

The class ids are wired to the backend in `app/services/anomaly_detection.py`
(`ANOMALY_CLASSES`), so `data.yaml` must list the classes in this exact order:

```yaml
path: <abs path to datasets/cctv>
train: train/images
val: val/images
nc: 4
names:
    0: kemacetan
    1: pohon_tumbang
    2: konstruksi
    3: kecelakaan
```

### Sourcing a dataset

There is no single public dataset covering all four classes, so the practical
route is to merge exports from Roboflow Universe (each free tier offers a
download as YOLOv11 format) into one project, then re-label/rename classes to
the four above. Good starting points found on Roboflow Universe:

| Class | Dataset | Notes |
|---|---|---|
| kemacetan | [Traffic Congestion Detection](https://universe.roboflow.com/sxc/traffic-congestion-detection) | ~150 street-camera images, vehicle classes + density |
| kemacetan | Traffic Jam Recognition (~4k images) | classification-style congestion vs. clear |
| pohon_tumbang | [Fallen Trees On Road](https://universe.roboflow.com/fallen-tree-on-roads/fallen-trees-on-road) | ~230–550 images of trees/obstructions on roads |
| konstruksi | [Work Zone Object Detection](https://universe.roboflow.com/18744-step2/work-zone-object-detection-du0gk) | ~3.5k images of work-zone boundaries & signs |
| konstruksi | Roadwork Object Detection (~180 images) | smaller fallback |
| kecelakaan | [Accident Detection Model](https://universe.roboflow.com/traffic-u05r8/accident-detection-model-6rv64) | ~3.3k images, single `Accident` class incl. negative frames |
| kecelakaan | Traffic Accident Detection (~1k images) | smaller fallback |

For a research-grade multi-class benchmark, the
[DoTA dataset](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)
(~4.7k clips, 18 anomaly classes) is excellent — but it is dashcam footage,
not fixed municipal CCTV, so expect a domain gap with the Semarang cameras.

After merging: export each project in YOLO format, drop or map their class
names onto the 4 ids above, and keep a clean train/val split.

### Train

Point `--data` at a `data.yaml` **or** straight at the dataset folder:

```bash
# Nano model (recommended for CPU), 100 epochs
uv run python -m scripts.train_yolo --data datasets/cctv

# Small model, more epochs
uv run python -m scripts.train_yolo --data datasets/cctv --model yolo11s.pt --epochs 200

# Tweak CPU training: smaller images + fixed small batch
uv run python -m scripts.train_yolo --data datasets/cctv --imgsz 416 --batch 8 --device cpu

# Resume an interrupted run
uv run python -m scripts.train_yolo --data datasets/cctv --resume
```

Results (weights, plots, metrics) are written to `runs/detect/<name>/` —
`best.pt` is the model to use for detection/inference.

### Train on Google Colab (free T4 GPU — recommended for full runs)

CPU training works but is slow (a 100-epoch run takes ~20+ hours). For the
real-quality run, use the included Colab notebook — it mirrors
`scripts/train_yolo.py` but trains on Colab's free **T4 GPU** (~20–40 min for
100 epochs), then you download `best.pt` and run it on CPU locally:

1. Open [`notebooks/train_yolo_colab.ipynb`](notebooks/train_yolo_colab.ipynb)
   in Colab (colab.research.google.com → **File → Upload notebook**), or if this
   repo is on GitHub:
   `https://colab.research.google.com/github/<user>/<repo>/blob/main/notebooks/train_yolo_colab.ipynb`
2. **Runtime → Change runtime type → T4 GPU**
3. Run all cells; upload your dataset zip (Roboflow export) when prompted
4. Download `best.pt` and copy it into the project:
   `models/best.pt`

The notebook auto-detects your split layout, re-emits `data.yaml` with absolute
paths (avoids Roboflow `../` relative-path quirks), reports final mAP metrics,
and zips up the weights + plots for download. The model is hardware-agnostic —
GPU training, CPU inference.

---

## Anomaly-aware route ranking

`POST /api/routes` returns the 3 best routes between two points (TomTom
Routing API), ranked **anomaly-weighted**: one frame is grabbed from every
active CCTV camera near each route and run through the trained anomaly model
(`ANOMALY_MODEL_PATH`, default `models/best.pt`). The four detected classes
and their server-side confidence thresholds live in
`app/services/anomaly_detection.py` (`ANOMALY_CLASSES`): kemacetan 0.35,
pohon_tumbang 0.40, konstruksi 0.40, kecelakaan 0.45. Anomaly analysis always
runs.

```json
{
  "origin": {"lat": -6.9867, "lng": 110.4139},
  "destination": {"lat": -6.9846, "lng": 110.3835}
}
```

Only `origin` and `destination` are required — `threshold_m` (150 m corridor)
and the per-class anomaly confidences are fixed server-side and cannot be
customized.

Response highlights — the response is typed by the `RouteData` model; camera
ids, owners, statuses are **not** exposed — each anomaly carries the camera's
location name, coordinates, type, confidence, and stream URL (so the frontend
can show the live feed of the incident). Each route also carries `guidance`: turn-by-turn maneuver
instructions from TomTom in Indonesian (`message`, `maneuver` type, street
name, and the maneuver `point` coordinates — everything a navigation UI needs):

```jsonc
{
  "success": true,
  "data": {
    "recommended_route_index": 0,
    "routes": [
      {
        "index": 0,
        "score": 100.0,                    // anomaly-weighted; best route = 100
        "recommended": true,
        "guidance": [                       // turn-by-turn instructions (id-ID)
          {
            "type": "LOCATION_DEPARTURE", "maneuver": "DEPART",
            "message": "Berangkat dari Jalan Mgr Sugiopranoto/14",
            "street": "Jalan Mgr Sugiopranoto",
            "point": {"lat": -6.98399, "lng": 110.40869},
            "route_offset_in_meters": 0, "travel_time_in_seconds": 0
          },
          {
            "type": "TURN", "maneuver": "TURN_LEFT",
            "message": "Belok kiri ke Jalan Simpang Lima",
            "street": "Jalan Simpang Lima",
            "point": {"lat": -6.98975, "lng": 110.4224}
          }
        ],
        "anomalies": [                      // only anomalies: type + location + confidence
          {
            "name": "KYAI SALEH",
            "latitude": -6.9867, "longitude": 110.4139,
            "anomaly_type": "kemacetan", "label": "Kemacetan",
            "confidence": 0.82, "count": 3
          },
          {
            "name": "SIMPANG LIMA 1 360",
            "latitude": -6.9894, "longitude": 110.4224,
            "anomaly_type": "kecelakaan", "label": "Kecelakaan",
            "confidence": 0.71, "count": 1
          }
        ]
      }
    ]
  }
}
```

How scoring works:

- Each camera's frame runs through the model; detections above the class's own
  threshold count as anomalies. A camera can report several classes at once.
  Each detected anomaly appears in `anomalies` with its type, Indonesian
  label, confidence, box count, and exact coordinates.
- Internally, each route gets an `anomaly_risk` (0–1) combining anomalous
  camera coverage (60%) with the most severe detection — severity × confidence
  (40%). Per-class severity: kecelakaan 1.0 (roadblock), pohon_tumbang 1.0
  (roadblock), konstruksi 0.6 (partial block), kemacetan 0.4 (slowdown).
- Combined cost = `travel_time * (1 + 2 * anomaly_risk)` — a route with a
  road-blocking anomaly is penalized as if it took ~3× longer. The cheapest
  route scores 100 and is marked `recommended`.
- Cameras appearing on several routes are analyzed only once; frames are
  grabbed in parallel (`ANOMALY_MAX_WORKERS`, default 8) while inference is
  serialized (ultralytics models are not thread-safe). Dead/offline streams are
  skipped and never fail the request.

Live analysis is network-bound (~2–5 s per camera), so a request checking 10–30
cameras takes tens of seconds. Relevant env vars: `ANOMALY_MODEL_PATH`,
`ANOMALY_IMGSZ`, `ANOMALY_GRAB_TIMEOUT_MS`, `ANOMALY_MAX_WORKERS`
(`FLOOD_MODEL_PATH` is still honored as a fallback for the model path).

---

## Run with Docker

```bash
docker compose up --build -d
```

- All endpoints live under `/api` (e.g. `POST /api/routes`). Docs:
  http://localhost:9002/docs — Health: `http://localhost:9002/api/health`
- Env vars come from `.env` (see `.env.example`); `TOM_API_KEY` is required for
  routing. The in-app scheduler runs the daily CCTV scrape inside the container
  (set `SCHEDULER_ENABLED=false` in `.env` to disable it).
- **No database is needed at runtime** — SQLAlchemy/alembic are migration-only
  here. The `cctvs_data` named volume mounts the container's `/app/data`
  directory (Docker volumes are directories — they can't mount onto the single
  file `cctvs.json`), so the scraper's daily updates persist across rebuilds.
  On first run Docker seeds the volume from the image's `data/cctvs.json`
  (`docker compose exec api cat /app/data/cctvs.json` to inspect).
- The trained model `models/best.pt` is **committed to the repo** (every other
  `*.pt` is gitignored) and baked into the image at `models/best.pt` (the
  `ANOMALY_MODEL_PATH` default), so the build works on any machine that clones
  the repo. To use your own weights, drop a new `best.pt` into `./models/` and
  uncomment the model volume in `docker-compose.yml`.
- The image installs **CPU-only torch** (from the PyTorch CPU index) to keep it
  ~2 GB smaller than the default CUDA wheels; the pinned torch version must
  exist on `download.pytorch.org/whl/cpu` (it does for stable releases).

On a fresh device:

```bash
git clone <repo-url> && cd SemarangVision-BE
cp .env.example .env        # then fill in TOM_API_KEY (and any DB settings)
docker compose up --build -d
```

`data/cctvs.json` is also tracked and seeded into the image (the scraper then
keeps it fresh in the `cctvs_data` volume). Stop with `docker compose down`
(add `-v` to also remove the volume — the next `up` re-seeds it from the
image). If you ever see the `not a directory` mount error, remove the stale
volume first: `docker compose down -v`.
