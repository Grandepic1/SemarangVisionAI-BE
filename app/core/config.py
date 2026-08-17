import os

from dotenv import load_dotenv
from sqlalchemy.engine import URL

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "Semarang Vision AI")

# --- Scheduler settings ---
# Set to "false" to disable the in-app daily scrape job (e.g. during dev/testing).
SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"
# Daily scrape timezone. Asia/Jakarta = WIB (UTC+7).
SCRAPE_TIMEZONE = os.getenv("SCRAPE_TIMEZONE", "Asia/Jakarta")

# --- Database settings (component-based; no full URL in env) ---
DB_TYPE = os.getenv("DB_TYPE", "postgresql").lower()  # "postgresql" or "mysql"
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "")
DB_USER = os.getenv("DB_USER", "root" if DB_TYPE == "mysql" else "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "semarangvision")

# --- TOM API KEY ---
TOM_API_KEY = os.getenv("TOM_API_KEY")

# --- Anomaly detection (kemacetan, pohon_tumbang, konstruksi, kecelakaan) ---
# Path to the trained YOLO anomaly model. Relative paths resolve against the
# project root (the trained weights live in models/ by default). The legacy
# FLOOD_MODEL_PATH name is still honored as a fallback during the transition.
ANOMALY_MODEL_PATH = os.getenv("ANOMALY_MODEL_PATH") or os.getenv("FLOOD_MODEL_PATH", "models/best.pt")
# Inference image size passed to the model.
ANOMALY_IMGSZ = int(os.getenv("ANOMALY_IMGSZ", "640"))
# Device for anomaly inference: "auto" uses the first CUDA GPU when available
# (falls back to CPU), or force a specific device like "0" or "cpu".
ANOMALY_DEVICE = os.getenv("ANOMALY_DEVICE", "auto")
# Per-camera frame-grab timeout in milliseconds (open + first read).
ANOMALY_GRAB_TIMEOUT_MS = int(os.getenv("ANOMALY_GRAB_TIMEOUT_MS", "10000"))
# How many camera streams are grabbed/detected concurrently.
ANOMALY_MAX_WORKERS = int(os.getenv("ANOMALY_MAX_WORKERS", "8"))

# --- Multi-frame anomaly confirmation (B+C+D rules) ---
# Each camera is sampled ANOMALY_SAMPLE_FRAMES times, ANOMALY_SAMPLE_INTERVAL_S
# seconds apart, and an anomaly is reported only when the evidence holds up:
#  - persistence: the class must appear in >= ANOMALY_MIN_FRACTION of samples
#  - kecelakaan:  the box region must show motion (an event), suppressing the
#    classic "parked cars read as accident" false positive
#  - konstruksi:  suppressed when traffic flows freely through the boxes unless
#    congestion is also detected (a work zone either jams traffic or closes the
#    road; cones with cars flowing past are just lane guidance)
ANOMALY_SAMPLE_FRAMES = int(os.getenv("ANOMALY_SAMPLE_FRAMES", "3"))
ANOMALY_SAMPLE_INTERVAL_S = float(os.getenv("ANOMALY_SAMPLE_INTERVAL_S", "5"))
ANOMALY_MIN_FRACTION = float(os.getenv("ANOMALY_MIN_FRACTION", "0.6"))
# Fraction of changed pixels (absdiff > 15) inside a class's box region that
# counts as "motion". kecelakaan needs >= EVENT_MOTION; konstruksi is
# suppressed when motion >= FLOW_MOTION (and no congestion is present).
ANOMALY_EVENT_MOTION = float(os.getenv("ANOMALY_EVENT_MOTION", "0.02"))
ANOMALY_FLOW_MOTION = float(os.getenv("ANOMALY_FLOW_MOTION", "0.05"))

_DRIVER_BY_TYPE = {
    "postgresql": ("postgresql+psycopg2", "5432"),
    "mysql": ("mysql+pymysql", "3306"),
}


def build_database_url() -> str:
    """Build a SQLAlchemy URL from the individual DB settings in env.

    URL.create() handles escaping of special characters (e.g. "@" or ":"
    inside the password) correctly, unlike manual string formatting.
    """
    if DB_TYPE not in _DRIVER_BY_TYPE:
        supported = ", ".join(_DRIVER_BY_TYPE)
        raise ValueError(f"Unsupported DB_TYPE '{DB_TYPE}'. Supported types: {supported}")

    driver, default_port = _DRIVER_BY_TYPE[DB_TYPE]
    port = int(DB_PORT or default_port)
    return URL.create(
        drivername=driver,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=port,
        database=DB_NAME,
    ).render_as_string(hide_password=False)


DATABASE_URL = build_database_url()
