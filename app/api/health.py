"""Health endpoint: lives under the global /api prefix as GET /api/health."""

from fastapi import APIRouter

from app.utils.response import ApiResponse, success

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=ApiResponse)
async def health():
    return success({"message": "API is running"}, "Success")
