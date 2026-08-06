import asyncio

from fastapi import APIRouter

from app.models.tomtom import RouteRequest
from app.services.tomtom import TomTomError, get_route_cameras
from app.utils.response import ApiResponse, fail, success

router = APIRouter(prefix="/tomtom", tags=["TomTom"])


@router.post("/routes", response_model=ApiResponse)
async def find_route_cameras(request: RouteRequest):
    """Calculate the 3 best routes between two points and list the CCTV cameras on them."""
    try:
        data = await asyncio.to_thread(
            get_route_cameras,
            (request.origin.lat, request.origin.lng),
            (request.destination.lat, request.destination.lng),
            request.threshold_m,
        )
    except TomTomError as exc:
        return fail(str(exc))
    return success(data, "Routes calculated")
