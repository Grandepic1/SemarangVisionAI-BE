import os

from dotenv import load_dotenv
from fastapi import FastAPI

from app.utils.response import ApiResponse, success

load_dotenv()
app = FastAPI(
    title=os.getenv("APP_NAME"),
    version="0.1.0"
)

@app.get("/", response_model=ApiResponse)
async def root():
    return success(
        {"message":"API is running"},
        "Success"
    )