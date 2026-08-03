from fastapi import FastAPI

from app.utils.response import ApiResponse, success

app = FastAPI(
    title="Bandung Vision AI",
    version="0.1.0"
)

@app.get("/", response_model=ApiResponse)
async def root():
    return success(
        {"message":"API is running"},
        "Success"
    )