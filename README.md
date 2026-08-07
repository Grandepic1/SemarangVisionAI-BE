### Bandung Vision AI (Backend)

FastAPI backend for the Bandung Vision CCTV project.

---

## Model Training (YOLO11)

Train a YOLO11 detector (`yolo11n` / `yolo11s`) on your CCTV dataset. Runs on
**CPU by default** — no GPU required (the CPU-only torch build is installed).

### Dataset format (YOLO)

```
datasets/cctv/
├── data.yaml            # optional — auto-generated if missing
├── train/
│   ├── images/          # *.jpg / *.png
│   └── labels/          # *.txt  (one line per object: class x y w h, normalized)
└── val/
    ├── images/
    └── labels/
```

`data.yaml` (auto-generated when you point at the folder):

```yaml
path: <abs path to datasets/cctv>
train: train/images
val: val/images
nc: 1
names:
    0: cctv
```

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
   `runs/detect/flood_yolo11s_run/best.pt`

The notebook auto-detects your split layout, re-emits `data.yaml` with absolute
paths (avoids Roboflow `../` relative-path quirks), reports final mAP metrics,
and zips up the weights + plots for download. The model is hardware-agnostic —
GPU training, CPU inference.

---

## Flood-aware route ranking

`POST /api/routes` returns the 3 best routes between two points (TomTom
Routing API), ranked **flood-weighted**: one frame is grabbed from every active
CCTV camera near each route and run through the trained flood model
(`FLOOD_MODEL_PATH`, default `flood_yolo11s_run/best.pt` — the completed Colab
run, mAP50 0.81). Flood analysis always runs.

```json
{
  "origin": {"lat": -6.9867, "lng": 110.4139},
  "destination": {"lat": -6.9846, "lng": 110.3835}
}
```

Only `origin` and `destination` are required — `threshold_m` (150 m corridor)
and `flood_confidence` (0.3) are fixed server-side and cannot be customized.

Response highlights — the response is typed by the `RouteData` model; camera
ids, stream URLs, owners, statuses are **not** exposed, only the location of
flooded cameras:

```jsonc
{
  "success": true,
  "data": {
    "recommended_route_index": 0,
    "routes": [
      {
        "index": 0,
        "score": 100.0,                    // flood-weighted; best route = 100
        "recommended": true,
        "floods": [                         // only floods: location name + where + confidence
          {"name": "KYAI SALEH", "latitude": -6.9867, "longitude": 110.4139, "flood_confidence": 0.87}
        ]
      }
    ]
  }
}
```

How scoring works:

- Each camera's frame runs through the model; detections above
  `flood_confidence` count as flooded. Each detected flood appears in
  `floods` with its exact coordinates (location name + lat/lng + confidence).
- Internally, each route gets a `flood_risk` (0–1) combining flooded-camera
  coverage (60%) with the max detection confidence (40%).
- Combined cost = `travel_time * (1 + 2 * flood_risk)` — a flooded route is
  penalized as if it took ~3× longer. The cheapest route scores 100 and is
  marked `recommended`.
- Cameras appearing on several routes are analyzed only once; frames are
  grabbed in parallel (`FLOOD_MAX_WORKERS`, default 8) while inference is
  serialized (ultralytics models are not thread-safe). Dead/offline streams are
  skipped and never fail the request.

Live analysis is network-bound (~2–5 s per camera), so a request checking 10–30
cameras takes tens of seconds. Relevant env vars: `FLOOD_MODEL_PATH`,
`FLOOD_IMGSZ`, `FLOOD_GRAB_TIMEOUT_MS`, `FLOOD_MAX_WORKERS`.

---

## Run with Docker

```bash
docker compose up --build -d
```

- All endpoints live under `/api` (e.g. `POST /api/routes`). Docs:
  http://localhost:8000/docs — Health: `http://localhost:8000/api/health`
- Env vars come from `.env` (see `.env.example`); `TOM_API_KEY` is required for
  routing. The in-app scheduler runs the daily CCTV scrape inside the container
  (set `SCHEDULER_ENABLED=false` in `.env` to disable it).
- **No database is needed at runtime** — SQLAlchemy/alembic are migration-only
  here. `data/cctvs.json` (the scraper's output) lives in the `cctvs_data`
  named volume so daily updates persist across rebuilds
  (`docker compose exec api cat /app/data/cctvs.json` to inspect).
- The trained model `runs/detect/flood_yolo11s_run/best.pt` is **committed to
  the repo** (gitignored otherwise) and baked into the image at
  `flood_yolo11s_run/best.pt` (the `FLOOD_MODEL_PATH` default), so the build
  works on any machine that clones the repo. To use your own weights, replace
  that file (or drop a `best.pt` into `./flood_yolo11s_run/` and uncomment the
  model volume in `compose.yaml`).
- The image installs **CPU-only torch** (from the PyTorch CPU index) to keep it
  ~2 GB smaller than the default CUDA wheels; the pinned torch version must
  exist on `download.pytorch.org/whl/cpu` (it does for stable releases).

On a fresh device:

```bash
git clone <repo-url> && cd BandungVision-BE
cp .env.example .env        # then fill in TOM_API_KEY (and any DB settings)
docker compose up --build -d
```

`data/cctvs.json` is also tracked and seeded into the image (the scraper then
keeps it fresh in the `cctvs_data` volume). Stop with `docker compose down`
(add `-v` to also remove volumes).
