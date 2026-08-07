from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.core.config import APP_NAME
from app.core.scheduler import shutdown_scheduler, start_scheduler
from app.api.tomtom import router as tomtom_router
from app.utils.exceptions import (
    AppException,
    app_exception_handler,
    validation_exception_handler,
)
from app.utils.response import ApiResponse, success


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    shutdown_scheduler()


app = FastAPI(
    title=APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

# Custom exceptions -> consistent {"success": false, ...} JSON error responses.
app.add_exception_handler(AppException, app_exception_handler)
# 422 validation errors use the same shape as other API errors.
app.add_exception_handler(RequestValidationError, validation_exception_handler)

app.include_router(tomtom_router)


@app.get("/", response_model=ApiResponse)
async def root():
    return success(
        {"message": "API is running"},
        "Success",
    )
