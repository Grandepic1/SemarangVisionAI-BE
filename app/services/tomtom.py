"""TomTom Routing integration.

Calculates the best routes between two coordinates and matches CCTV
cameras (from cctvs.json) that lie on/near each route, so downstream
processing can act on the cameras covering a trip.
"""

import math
from dataclasses import dataclass

import requests

from app.core.config import TOM_API_KEY
from app.models.tomtom import DEFAULT_THRESHOLD_M
from app.services.scraping import load_existing

ROUTING_URL = "https://api.tomtom.com/routing/1/calculateRoute/{origin}:{destination}/json"

# One degree of latitude ≈ 111.32 km (equirectangular projection reference).
_METERS_PER_DEG = 111_320.0


class TomTomError(Exception):
    """Raised when the TomTom API request fails."""


@dataclass
class Route:
    index: int
    length_in_meters: float
    travel_time_in_seconds: float
    traffic_delay_in_seconds: float
    points: list[tuple[float, float]]  # (latitude, longitude) along the polyline


def _check_api_key() -> None:
    if not TOM_API_KEY:
        raise TomTomError("TOM_API_KEY is not set. Add it to your .env file.")


def calculate_routes(
    origin: tuple[float, float],
    destination: tuple[float, float],
    max_alternatives: int = 2,
) -> list[Route]:
    """Request the best routes between two coordinates.

    With max_alternatives=2 TomTom returns 3 routes total: the fastest
    route plus two alternatives. Returns [] if no routes are found.
    """
    _check_api_key()

    origin_str = f"{origin[0]:.6f},{origin[1]:.6f}"
    dest_str = f"{destination[0]:.6f},{destination[1]:.6f}"

    try:
        response = requests.get(
            ROUTING_URL.format(origin=origin_str, destination=dest_str),
            params={
                "key": TOM_API_KEY,
                "maxAlternatives": max_alternatives,
                "routeType": "fastest",
                "travelMode": "car",
                "computeTravelTimeFor": "all",
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise TomTomError(f"TomTom Routing API request failed: {exc}") from exc
    if response.status_code != 200:
        raise TomTomError(
            f"TomTom Routing API error {response.status_code}: {response.text[:300]}"
        )

    routes = []
    for index, raw in enumerate(response.json().get("routes", [])):
        summary = raw.get("summary", {})
        points: list[tuple[float, float]] = []
        for leg in raw.get("legs", []):
            for point in leg.get("points", []):
                points.append((point["latitude"], point["longitude"]))
        routes.append(
            Route(
                index=index,
                length_in_meters=summary.get("lengthInMeters", 0),
                travel_time_in_seconds=summary.get("travelTimeInSeconds", 0),
                traffic_delay_in_seconds=summary.get("trafficDelayInSeconds", 0),
                points=points,
            )
        )
    return routes


def _point_segment_distance_m(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """Distance in meters from point to segment [a, b] (planar approximation).

    Uses an equirectangular projection around the point's latitude, which
    is accurate enough (< 0.1%) for the corridor distances we care about.
    """
    ref_lat = point[0]
    cos_lat = math.cos(math.radians(ref_lat))

    def project(p: tuple[float, float]) -> tuple[float, float]:
        return (p[1] * _METERS_PER_DEG * cos_lat, p[0] * _METERS_PER_DEG)

    px, py = project(point)
    ax, ay = project(a)
    bx, by = project(b)

    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - ax, py - ay)

    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_route_m(latitude: float, longitude: float, points: list[tuple[float, float]]) -> float:
    """Shortest distance in meters from a coordinate to a route polyline."""
    if not points:
        return math.inf
    if len(points) == 1:
        return _point_segment_distance_m((latitude, longitude), points[0], points[0])
    return min(
        _point_segment_distance_m((latitude, longitude), a, b)
        for a, b in zip(points, points[1:])
    )


def cameras_on_route(
    cameras: list[dict],
    route: Route,
    threshold_m: float = DEFAULT_THRESHOLD_M,
) -> list[dict]:
    """Return cameras within threshold_m of the route, annotated with their distance."""
    matched = []
    for cam in cameras:
        distance = distance_to_route_m(cam["latitude"], cam["longitude"], route.points)
        if distance <= threshold_m:
            matched.append({**cam, "distance_to_route_m": round(distance, 1)})
    return matched


def get_route_cameras(
    origin: tuple[float, float],
    destination: tuple[float, float],
    threshold_m: float = DEFAULT_THRESHOLD_M,
    cameras: list[dict] | None = None,
) -> dict:
    """Calculate the best routes and match CCTV cameras on each route.

    Returns a dict shaped for the API response:
    {
      "origin": {"lat": ..., "lng": ...},
      "destination": {"lat": ..., "lng": ...},
      "threshold_m": 150.0,
      "routes": [
        {
          "index": 0,
          "length_in_meters": ...,
          "travel_time_in_seconds": ...,
          "traffic_delay_in_seconds": ...,
          "points": [[lat, lng], ...],
          "cameras_on_route": [...],
        }, ...
      ],
    }
    """
    routes = calculate_routes(origin, destination)
    if cameras is None:
        cameras = load_existing()

    route_payloads = []
    for route in routes:
        matched = cameras_on_route(cameras, route, threshold_m)

        # TODO(process): cameras_on_route input untuk mengecek semua camera yang ada di rute
        route_payloads.append(
            {
                "index": route.index,
                "length_in_meters": route.length_in_meters,
                "travel_time_in_seconds": route.travel_time_in_seconds,
                "traffic_delay_in_seconds": route.traffic_delay_in_seconds,
                "points": [[lat, lng] for lat, lng in route.points],
                "cameras_on_route": matched,
            }
        )

    return {
        "origin": {"lat": origin[0], "lng": origin[1]},
        "destination": {"lat": destination[0], "lng": destination[1]},
        "threshold_m": threshold_m,
        "routes": route_payloads,
    }
