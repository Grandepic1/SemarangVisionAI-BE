import asyncio

from fastapi import APIRouter

from app.models.tomtom import RouteData, RouteRequest
from app.services.tomtom import get_route_cameras
from app.utils.response import ApiResponse, ErrorResponse, success

# NOTE: the /api prefix is applied globally in app.main (api_router) —
# routers must not hardcode it.
router = APIRouter(tags=["Route"])


@router.post(
    "/routes",
    response_model=ApiResponse[RouteData],
    responses={
        # All error paths share the ApiResponse envelope with success=false
        # and data=null (see the app-level exception handlers in main.py).
        401: {
            "model": ErrorResponse,
            "description": "TOM_API_KEY is not set. Add it to the .env file.",
        },
        422: {
            "model": ErrorResponse,
            "description": (
                "Request validation failed: malformed JSON, missing fields, "
                "or out-of-range values."
            ),
        },
        502: {
            "model": ErrorResponse,
            "description": "TomTom Routing API unreachable or returned an error.",
        },
    },
)
async def find_route_cameras(request: RouteRequest):
    """Calculate the 3 best routes between two points, ranked by flood risk.

    Errors (missing API key, TomTom upstream failures) raise AppException
    subclasses, which the app-level exception handler turns into
    {"success": false, ...} responses with proper HTTP status codes.
    """
    # threshold_m (150 m) and flood_confidence (0.3) are fixed server-side
    # defaults in the service — the frontend only provides coordinates.
    data = await asyncio.to_thread(
        get_route_cameras,
        (request.origin.lat, request.origin.lng),
        (request.destination.lat, request.destination.lng),
    )
    return success(data, "Routes calculated")
