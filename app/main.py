from fastapi import FastAPI

app = FastAPI(
    title="Bandung Vision AI",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {
        "message":"API is running"
    }