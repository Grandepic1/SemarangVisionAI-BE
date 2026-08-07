"""API routers, aggregated under the global ``/api`` prefix.

``app.main`` mounts :data:`api_router` with ``prefix="/api"``, so feature
routers must NOT hardcode ``/api`` themselves — any router added here is
automatically rooted at ``/api``.
"""

from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.tomtom import router as tomtom_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(tomtom_router)
