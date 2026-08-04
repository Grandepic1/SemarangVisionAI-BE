from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import APP_NAME
from app.core.scheduler import shutdown_scheduler, start_scheduler
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


@app.get("/", response_model=ApiResponse)
async def root():
    return success(
        {"message": "API is running"},
        "Success",
    )
