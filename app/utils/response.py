from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    message: str
    data: T | None = None


def success(data: Any = None, message: str = "Success"):
    return ApiResponse(success=True, message=message, data=data)


class ErrorResponse(ApiResponse[None]):
    """API error envelope: success=false with data=null.

    Used to document 4xx/5xx responses in OpenAPI; the app-level exception
    handlers (app.utils.exceptions) produce this exact shape. The literal
    type makes the OpenAPI schema render ``const: false`` so docs show
    success=false instead of Swagger's generic boolean sample (true).
    """

    success: Literal[False] = False


def fail(message: str):
    return ApiResponse(success=False, message=message, data=None)
