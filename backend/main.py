from fastapi import FastAPI

app = FastAPI(title="Project Metis Backend")


@app.get("/", tags=["root"])
async def read_root():
    return {"message": "Project Metis backend is running"}


@app.get("/health", tags=["health"])
async def health():
    """Simple health-check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    # Allow quick local run: `python backend/main.py` (works if uvicorn is installed)
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
