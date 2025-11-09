from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.chat import router as chat_router

app = FastAPI(title="Project Metis Backend")

# Allow frontend dev server (Vite default 5173) and local origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "*",  # adjust or remove wildcard in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


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
