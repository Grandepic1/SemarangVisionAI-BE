from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    message: str
    data: T | None = None


def success(data: Any = None, message: str = "Success"):
    return ApiResponse(success=True, message=message, data=data)


def fail(message: str):
    return ApiResponse(success=False, message=message, data=None)
