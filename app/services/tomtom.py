"""TomTom Routing integration.

Calculates the best routes between two coordinates and matches CCTV
cameras (from cctvs.json) that lie on/near each route, so downstream
processing can act on the cameras covering a trip.
"""

import math
from dataclasses import dataclass

import requests

from app.core.config import TOM_API_KEY
from app.models.tomtom import DEFAULT_FLOOD_CONFIDENCE, DEFAULT_THRESHOLD_M
from app.services.flood_detection import analyze_cameras, camera_key
from app.services.scraping import load_existing
from app.utils.exceptions import AppException

ROUTING_URL = "https://api.tomtom.com/routing/1/calculateRoute/{origin}:{destination}/json"

# One degree of latitude ≈ 111.32 km (equirectangular projection reference).
_METERS_PER_DEG = 111_320.0


class TomTomError(AppException):
    """Raised when the TomTom API request fails.

    Carries an HTTP status code so the FastAPI AppException handler can turn
    it into a proper error response: 401 for a missing API key, 502 when the
    upstream TomTom Routing API cannot be reached or errors out.
    """


@dataclass
class Route:
    index: int
    length_in_meters: float
    travel_time_in_seconds: float
    traffic_delay_in_seconds: float
    points: list[tuple[float, float]]  # (latitude, longitude) along the polyline


def _check_api_key() -> None:
    if not TOM_API_KEY:
        raise TomTomError(401, "TOM_API_KEY is not set. Add it to your .env file.")


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
        raise TomTomError(502, f"TomTom Routing API request failed: {exc}") from exc
    if response.status_code != 200:
        raise TomTomError(
            502,
            f"TomTom Routing API error {response.status_code}: {response.text[:300]}",
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


def _flood_risk(matched_cameras: list[dict]) -> float:
    """Route-level flood risk in 0..1, used only for scoring (not exposed).

    Combines how much of the route is flooded (coverage, 60%) with how
    confident the detections were (severity, 40%).
    """
    checked = [c for c in matched_cameras if c.get("flood_checked")]
    flooded = [c for c in checked if c.get("flood_detected")]
    if not checked:
        return 0.0
    max_confidence = max((float(c.get("flood_confidence") or 0.0) for c in flooded), default=0.0)
    coverage = len(flooded) / len(checked)
    return round(0.6 * coverage + 0.4 * max_confidence, 4)


def _score_routes(route_payloads: list[dict], flood_risks: list[float]) -> None:
    """Score routes in place: flood-weighted rank, best route gets 100.

    Combined cost = travel_time * (1 + 2 * flood_risk), so a route with a
    flood on it is penalized as if it took roughly 3x longer; the cheapest
    combined cost wins and is marked recommended.
    """
    if not route_payloads:
        return

    combined_costs = []
    for payload, flood_risk in zip(route_payloads, flood_risks):
        travel_time = float(payload["travel_time_in_seconds"] or 0.0)
        combined_costs.append(travel_time * (1 + 2 * flood_risk))

    best_cost = min(combined_costs)
    best_index = combined_costs.index(best_cost)  # first route wins ties

    for i, (payload, cost) in enumerate(zip(route_payloads, combined_costs)):
        payload["score"] = round(100.0 * best_cost / cost, 1) if cost > 0 else 0.0
        payload["recommended"] = i == best_index


def _merge_flood_results(route_payloads: list[dict], analyzed: dict) -> None:
    """Attach per-camera flood results back onto each route's camera list.

    The analyzed dict is shared across routes, so the route-specific
    distance_to_route_m annotation is restored from this route's own entry.
    """
    for payload in route_payloads:
        merged = []
        for cam in payload["cameras_on_route"]:
            result = analyzed.get(camera_key(cam))
            if result is None:
                result = {
                    **cam,
                    "flood_checked": False,
                    "flood_detected": False,
                    "flood_confidence": 0.0,
                    "flood_count": 0,
                }
            else:
                result = {**result, "distance_to_route_m": cam.get("distance_to_route_m")}
            merged.append(result)
        payload["cameras_on_route"] = merged


def get_route_cameras(
    origin: tuple[float, float],
    destination: tuple[float, float],
    threshold_m: float = DEFAULT_THRESHOLD_M,
    cameras: list[dict] | None = None,
    flood_confidence: float = DEFAULT_FLOOD_CONFIDENCE,
) -> dict:
    """Calculate the best routes, match CCTV cameras, and score them by flood risk.

    One frame is grabbed from every active camera on each route and run through
    the flood model; routes are then scored flood-weighted and the best one is
    marked ``recommended``. The response exposes only the *location* of flooded
    cameras (name + coordinates + confidence) — no ids, stream URLs, owners, or
    statuses leak out.

    Returns a dict shaped for the API response (see RouteData):
    {
      "origin": {"lat": ..., "lng": ...},
      "destination": {"lat": ..., "lng": ...},
      "threshold_m": 150.0,
      "recommended_route_index": 0,
      "routes": [
        {
          "index": 0,
          "length_in_meters": ...,
          "travel_time_in_seconds": ...,
          "traffic_delay_in_seconds": ...,
          "points": [[lat, lng], ...],
          "floods": [{"name", "latitude", "longitude", "flood_confidence"}, ...],
          "score": 100.0,
          "recommended": true,
        }, ...
      ],
    }
    """
    routes = calculate_routes(origin, destination)
    if cameras is None:
        cameras = load_existing()

    route_payloads = []
    all_matched = []
    for route in routes:
        matched = cameras_on_route(cameras, route, threshold_m)
        all_matched.extend(matched)
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

    # Analyze each unique active camera once, even if it appears on several routes.
    active = [c for c in all_matched if c.get("status") == "ACTIVE"]
    analyzed = analyze_cameras(active, flood_confidence)
    _merge_flood_results(route_payloads, analyzed)

    flood_risks = []
    for payload in route_payloads:
        flooded = [
            c
            for c in payload["cameras_on_route"]
            if c.get("flood_detected") and c.get("flood_checked")
        ]
        payload["floods"] = [
            {
                "name": c.get("location_name") or c.get("camera_name") or "Unknown",
                "latitude": c.get("latitude"),
                "longitude": c.get("longitude"),
                "flood_confidence": c.get("flood_confidence"),
            }
            for c in flooded
        ]
        flood_risks.append(_flood_risk(payload["cameras_on_route"]))
        del payload["cameras_on_route"]

    _score_routes(route_payloads, flood_risks)

    recommended_index = next(
        (p["index"] for p in route_payloads if p.get("recommended")),
        None,
    )
    return {
        "origin": {"lat": origin[0], "lng": origin[1]},
        "destination": {"lat": destination[0], "lng": destination[1]},
        "threshold_m": threshold_m,
        "recommended_route_index": recommended_index,
        "routes": route_payloads,
    }
