"""Pydantic request/response models for the TomTom API routes."""

from pydantic import BaseModel, Field

# Corridor width (meters) around a route that counts a camera as "on the route".
# Shared contract between the API layer (Field default) and the service layer.
DEFAULT_THRESHOLD_M = 150.0


class Coordinate(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class RouteRequest(BaseModel):
    origin: Coordinate
    destination: Coordinate
    # Corridor width (meters) around each route that counts a camera as "on route".
    threshold_m: float = Field(DEFAULT_THRESHOLD_M, ge=1, le=5000)
