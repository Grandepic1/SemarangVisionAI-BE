from fastapi import FastAPI

from app.core.config import APP_NAME
from app.utils.response import ApiResponse, success

app = FastAPI(
    title=APP_NAME,
    version="0.1.0",
)


@app.get("/", response_model=ApiResponse)
async def root():
    return success(
        {"message": "API is running"},
        "Success",
    )
