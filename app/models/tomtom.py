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


class RouteInfo(BaseModel):
    """One calculated route with its flood-weighted score."""

    index: int
    length_in_meters: float
    travel_time_in_seconds: float
    traffic_delay_in_seconds: float
    points: list[list[float]]  # [lat, lng] polyline
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
