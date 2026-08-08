"""Pydantic request/response models for the TomTom API routes."""

from pydantic import BaseModel, Field

# Fixed backend parameters — the frontend does not customize these.
# Corridor width (meters) around a route that counts a camera as "on the route".
DEFAULT_THRESHOLD_M = 150.0

# Minimum confidence for a YOLO detection to count as a flood.
DEFAULT_FLOOD_CONFIDENCE = 0.30


class Coordinate(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class RouteRequest(BaseModel):
    """Request body for POST /api/routes.

    threshold_m and flood_confidence are fixed server-side (150 m / 0.3);
    extra fields sent by the client are ignored.
    """

    origin: Coordinate
    destination: Coordinate


class FloodLocation(BaseModel):
    """A flood detected on the route: location name, coordinates, confidence."""

    name: str
    latitude: float
    longitude: float
    flood_confidence: float
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
    """One calculated route with its flood-weighted score."""

    index: int
    length_in_meters: float
    travel_time_in_seconds: float
    traffic_delay_in_seconds: float
    points: list[list[float]]  # [lat, lng] polyline
    guidance: list[GuidanceInstruction]
    floods: list[FloodLocation]
    score: float
    recommended: bool


class RouteData(BaseModel):
    """Response body for POST /api/routes."""

    origin: Coordinate
    destination: Coordinate
    threshold_m: float
    recommended_route_index: int | None
    routes: list[RouteInfo]
