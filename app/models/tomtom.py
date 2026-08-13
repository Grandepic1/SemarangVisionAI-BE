"""Pydantic request/response models for the TomTom API routes."""

from pydantic import BaseModel, Field

# Fixed backend parameters — the frontend does not customize these.
# Corridor width (meters) around a route that counts a camera as "on the route".
DEFAULT_THRESHOLD_M = 150.0

# Per-class anomaly confidence thresholds live in app.services.anomaly_detection
# (ANOMALY_CLASSES) — rare, serious classes need higher evidence than common ones.


class Coordinate(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class RouteRequest(BaseModel):
    """Request body for POST /api/routes.

    threshold_m and the per-class anomaly confidences are fixed server-side
    (150 m corridor, thresholds in ANOMALY_CLASSES); extra fields sent by the
    client are ignored.
    """

    origin: Coordinate
    destination: Coordinate


class AnomalyEvent(BaseModel):
    """One anomaly detected on the route: location, type, and confidence.

    A single camera can contribute several events (one per detected class).
    anomaly_type is the machine key (kemacetan / pohon_tumbang / konstruksi /
    kecelakaan); label is the human-readable Indonesian name.
    """

    name: str
    latitude: float
    longitude: float
    anomaly_type: str
    label: str
    confidence: float
    count: int = 1
    stream_url: str | None = None


class GuidanceInstruction(BaseModel):
    """One turn-by-turn maneuver instruction for a route.

    Field names mirror TomTom's guidance.instructions[] shape, converted
    to snake_case; everything the frontend needs to render a step is
    optional except the message itself.
    """

    type: str  # instructionType: TURN, LOCATION_DEPARTURE, LOCATION_ARRIVAL, ...
    maneuver: str  # DEPART, TURN_LEFT, ROUNDABOUT_CROSS, ARRIVE, ...
    message: str  # localized instruction text, e.g. "Belok kiri ke Jalan Simpang Lima"
    street: str | None = None
    road_numbers: list[str] | None = None
    point: Coordinate | None = None  # maneuver coordinates (lat/lng)
    route_offset_in_meters: float = 0
    travel_time_in_seconds: float = 0
    roundabout_exit_number: int | None = None
    turn_angle_in_decimal_degrees: float | None = None


class RouteInfo(BaseModel):
    """One calculated route with its anomaly-weighted score."""

    index: int
    length_in_meters: float
    travel_time_in_seconds: float
    traffic_delay_in_seconds: float
    points: list[list[float]]  # [lat, lng] polyline
    guidance: list[GuidanceInstruction]
    anomalies: list[AnomalyEvent]
    score: float
    recommended: bool


class RouteData(BaseModel):
    """Response body for POST /api/routes."""

    origin: Coordinate
    destination: Coordinate
    threshold_m: float
    recommended_route_index: int | None
    routes: list[RouteInfo]
