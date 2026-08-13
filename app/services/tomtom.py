"""TomTom Routing integration.

Calculates the best routes between two coordinates and matches CCTV
cameras (from data/cctvs.json) that lie on/near each route, so downstream
processing can act on the cameras covering a trip.
"""

import math
from dataclasses import dataclass, field

import requests

from app.core.config import TOM_API_KEY
from app.models.tomtom import DEFAULT_THRESHOLD_M
from app.services.anomaly_detection import analyze_cameras, camera_key
from app.services.scraping import load_existing
from app.utils.exceptions import AppException

ROUTING_URL = "https://api.tomtom.com/routing/1/calculateRoute/{origin}:{destination}/json"

# Turn-by-turn instruction language. id-ID = Indonesian (TomTom's official
# localized instructions). Change to e.g. "en-US" for English.
ROUTING_LANGUAGE = "id-ID"

# One degree of latitude ≈ 111.32 km (equirectangular projection reference).
_METERS_PER_DEG = 111_320.0

# Per-class severity used for route risk scoring (0..1). Road-blocking
# anomalies (kecelakaan, pohon_tumbang) weigh more than slowdowns
# (kemacetan); konstruksi partially blocks the road.
_ANOMALY_SEVERITY = {
    "kecelakaan": 1.0,
    "pohon_tumbang": 1.0,
    "konstruksi": 0.6,
    "kemacetan": 0.4,
}


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
    guidance: list[dict] = field(default_factory=list)  # maneuver instructions (see _parse_guidance)


def _check_api_key() -> None:
    if not TOM_API_KEY:
        raise TomTomError(401, "TOM_API_KEY is not set. Add it to your .env file.")


def _parse_guidance(raw_route: dict) -> list[dict]:
    """Extract turn-by-turn maneuver instructions from a raw TomTom route.

    Returns a list of dicts shaped like the GuidanceInstruction model
    (snake_case fields, maneuver point as {lat, lng}), or [] if TomTom
    returned no guidance for the route.
    """
    instructions = raw_route.get("guidance", {}).get("instructions", [])
    parsed = []
    for inst in instructions:
        message = inst.get("message")
        if not message:
            # The response model requires instruction text; skip any entry
            # TomTom returned without a message instead of risking a 500.
            continue
        point = inst.get("point")
        parsed.append(
            {
                "type": inst.get("instructionType") or "",
                "maneuver": inst.get("maneuver") or "",
                "message": message,
                "street": inst.get("street"),
                "road_numbers": inst.get("roadNumbers"),
                "point": (
                    {"lat": point["latitude"], "lng": point["longitude"]}
                    if point
                    else None
                ),
                "route_offset_in_meters": inst.get("routeOffsetInMeters", 0),
                "travel_time_in_seconds": inst.get("travelTimeInSeconds", 0),
                "roundabout_exit_number": inst.get("roundaboutExitNumber"),
                "turn_angle_in_decimal_degrees": inst.get("turnAngleInDecimalDegrees"),
            }
        )
    return parsed


def calculate_routes(
    origin: tuple[float, float],
    destination: tuple[float, float],
    max_alternatives: int = 2,
) -> list[Route]:
    """Request the best routes between two coordinates.

    With max_alternatives=2 TomTom returns 3 routes total: the fastest
    route plus two alternatives. Returns [] if no routes are found.
    Each Route carries turn-by-turn guidance instructions in Indonesian
    (see ROUTING_LANGUAGE).
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
                "instructionsType": "text",
                "language": ROUTING_LANGUAGE,
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
                guidance=_parse_guidance(raw),
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


def _anomaly_risk(matched_cameras: list[dict]) -> float:
    """Route-level anomaly risk in 0..1, used only for scoring (not exposed).

    Combines how much of the route has an anomaly (coverage, 60%) with the
    most severe detection — severity * confidence, so an accident outranks a
    congestion (40%).
    """
    checked = [c for c in matched_cameras if c.get("anomaly_checked")]
    if not checked:
        return 0.0
    anomalous = [c for c in checked if c.get("has_anomaly")]
    if anomalous:
        worst = max(
            _ANOMALY_SEVERITY.get(a["type"], 0.5) * float(a["confidence"])
            for c in anomalous
            for a in c.get("anomalies", [])
        )
    else:
        worst = 0.0
    coverage = len(anomalous) / len(checked)
    return round(0.6 * coverage + 0.4 * worst, 4)


def _score_routes(route_payloads: list[dict], anomaly_risks: list[float]) -> None:
    """Score routes in place: anomaly-weighted rank, best route gets 100.

    Combined cost = travel_time * (1 + 2 * anomaly_risk), so a route with a
    road-blocking anomaly on it is penalized as if it took roughly 3x longer;
    the cheapest combined cost wins and is marked recommended.
    """
    if not route_payloads:
        return

    combined_costs = []
    for payload, anomaly_risk in zip(route_payloads, anomaly_risks):
        travel_time = float(payload["travel_time_in_seconds"] or 0.0)
        combined_costs.append(travel_time * (1 + 2 * anomaly_risk))

    best_cost = min(combined_costs)
    best_index = combined_costs.index(best_cost)  # first route wins ties

    for i, (payload, cost) in enumerate(zip(route_payloads, combined_costs)):
        payload["score"] = round(100.0 * best_cost / cost, 1) if cost > 0 else 0.0
        payload["recommended"] = i == best_index


def _merge_anomaly_results(route_payloads: list[dict], analyzed: dict) -> None:
    """Attach per-camera anomaly results back onto each route's camera list.

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
                    "anomaly_checked": False,
                    "anomalies": [],
                    "has_anomaly": False,
                    "max_confidence": 0.0,
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
) -> dict:
    """Calculate the best routes, match CCTV cameras, and score them by anomaly risk.

    One frame is grabbed from every active camera on each route and run through
    the anomaly model; routes are then scored anomaly-weighted and the best one
    is marked ``recommended``. The response exposes only the *location* of
    anomalous cameras (name + coordinates + anomaly type + confidence) plus the
    camera's stream URL (for the frontend's live view) — no ids, owners, or
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
          "guidance": [{"type", "maneuver", "message", "street", ...}, ...],
          "anomalies": [
            {"name", "latitude", "longitude", "anomaly_type", "label", "confidence", "count", "stream_url"},
            ...,
          ],
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
                "guidance": route.guidance,
                "cameras_on_route": matched,
            }
        )

    # Analyze each unique active camera once, even if it appears on several routes.
    active = [c for c in all_matched if c.get("status") == "ACTIVE"]
    analyzed = analyze_cameras(active)
    _merge_anomaly_results(route_payloads, analyzed)

    anomaly_risks = []
    for payload in route_payloads:
        anomalous = [
            c
            for c in payload["cameras_on_route"]
            if c.get("has_anomaly") and c.get("anomaly_checked")
        ]
        payload["anomalies"] = [
            {
                "name": c.get("location_name") or c.get("camera_name") or "Unknown",
                "latitude": c.get("latitude"),
                "longitude": c.get("longitude"),
                "anomaly_type": a["type"],
                "label": a["label"],
                "confidence": a["confidence"],
                "count": a["count"],
                "stream_url": c.get("stream_url"),
            }
            for c in anomalous
            for a in c.get("anomalies", [])
        ]
        anomaly_risks.append(_anomaly_risk(payload["cameras_on_route"]))
        del payload["cameras_on_route"]

    _score_routes(route_payloads, anomaly_risks)

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
