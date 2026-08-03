from fastapi import Request
from fastapi.responses import JSONResponse

from app.main import app


class AppException(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message

@app.exception_handler(AppException)
async def http_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.message,
            "data": None,
        },
    )