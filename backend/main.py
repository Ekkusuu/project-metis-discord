from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.chat import router as chat_router
from backend.llama_engine import get_config

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


@app.on_event("startup")
async def startup_event():
    """Run initialization tasks on backend startup."""
    config = get_config()
    
    # Preload the LLM model
    print("\n" + "="*60)
    print("Loading LLM model...")
    print("="*60)
    try:
        from backend.llama_engine import get_model
        model = get_model()
        print(f"✓ Model loaded successfully")
        print(f"  Model path: {config['model']['path']}")
        print(f"  Context size: {config['model'].get('n_ctx', 4096)}")
        print(f"  GPU layers: {config['model'].get('n_gpu_layers', -1)}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("="*60 + "\n")
    
    # RAG initialization
    rag_cfg = config.get("rag", {})
    if rag_cfg.get("enabled", False):
        print("="*60)
        print("RAG System: Checking for file changes...")
        print("="*60)
        
        try:
            from backend.rag_engine import index_all_folders, get_collection
            
            # Always check for changes (new, modified, or deleted files)
            collection = get_collection()
            count_before = collection.count()
            
            print(f"Current index: {count_before} chunks")
            print("Scanning for new, modified, or deleted files...")
            
            results = index_all_folders(clear_existing=False)
            
            count_after = collection.count()
            total_new_chunks = sum(results.values())
            
            if total_new_chunks > 0:
                print(f"\n✓ Indexed {total_new_chunks} new/updated chunks")
                print(f"  Total chunks in index: {count_before} → {count_after}")
            else:
                print(f"\n✓ No changes detected. Index is up to date.")
            
            print("="*60 + "\n")
        except Exception as e:
            print(f"Error during RAG auto-indexing: {e}")
            print("RAG system may not function correctly. Check your configuration.")


if __name__ == "__main__":
    # Allow quick local run: `python backend/main.py` (works if uvicorn is installed)
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
