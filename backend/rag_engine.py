from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings

from backend.llama_engine import get_config

# Global ChromaDB client
_chroma_client: Optional[chromadb.Client] = None
_collection: Optional[chromadb.Collection] = None

# File tracking metadata
_file_metadata_path: Optional[Path] = None


def get_file_metadata_path() -> Path:
    """Get the path to the file metadata tracking JSON."""
    global _file_metadata_path
    if _file_metadata_path is not None:
        return _file_metadata_path
    
    config = get_config()
    rag_cfg = config.get("rag", {})
    persist_dir = rag_cfg.get("persist_directory", ".chromadb")
    
    # Store metadata in the same directory as ChromaDB
    persist_path = Path(__file__).resolve().parents[1] / persist_dir
    persist_path.mkdir(parents=True, exist_ok=True)
    
    _file_metadata_path = persist_path / "file_metadata.json"
    return _file_metadata_path


def load_file_metadata() -> Dict[str, Dict[str, Any]]:
    """Load file metadata from disk."""
    metadata_path = get_file_metadata_path()
    if not metadata_path.exists():
        return {}
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file metadata: {e}")
        return {}


def save_file_metadata(metadata: Dict[str, Dict[str, Any]]) -> None:
    """Save file metadata to disk."""
    metadata_path = get_file_metadata_path()
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving file metadata: {e}")


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get file size and modification time."""
    try:
        stat = file_path.stat()
        return {
            "size": stat.st_size,
            "mtime": stat.st_mtime
        }
    except Exception:
        return {"size": 0, "mtime": 0}


def has_file_changed(file_path: Path, stored_metadata: Dict[str, Any]) -> bool:
    """Check if a file has changed based on size."""
    current_info = get_file_info(file_path)
    stored_info = stored_metadata.get(str(file_path), {})
    
    # If file wasn't indexed before, it's considered changed
    if not stored_info:
        return True
    
    # Check if size has changed
    return current_info["size"] != stored_info.get("size", 0)


def get_chroma_client() -> chromadb.Client:
    """Get or create the ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    config = get_config()
    rag_cfg = config.get("rag", {})
    persist_dir = rag_cfg.get("persist_directory", ".chromadb")

    # Resolve persist directory relative to project root
    persist_path = Path(__file__).resolve().parents[1] / persist_dir
    persist_path.mkdir(parents=True, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(
        path=str(persist_path),
        settings=Settings(anonymized_telemetry=False)
    )
    return _chroma_client


def get_collection() -> chromadb.Collection:
    """Get or create the RAG collection."""
    global _collection
    if _collection is not None:
        return _collection

    config = get_config()
    rag_cfg = config.get("rag", {})
    collection_name = rag_cfg.get("collection_name", "metis_knowledge")
    embedding_model = rag_cfg.get("embedding_model", "all-MiniLM-L6-v2")

    client = get_chroma_client()
    
    # Resolve embedding model path (relative to project root)
    model_path = Path(__file__).resolve().parents[1] / embedding_model
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Embedding model not found at: {model_path}\n"
            f"Expected folder: {embedding_model}\n"
            "Please download the model and place it in the project root."
        )
    
    # Load local sentence-transformers model
    from sentence_transformers import SentenceTransformer
    
    class LocalEmbeddingFunction:
        """Custom embedding function using local SentenceTransformer model."""
        def __init__(self, model_path: Path):
            self.model = SentenceTransformer(str(model_path), device='cpu')
            self._model_path = model_path
        
        def __call__(self, input: List[str]) -> List[List[float]]:
            """Embed a batch of texts."""
            embeddings = self.model.encode(input, convert_to_numpy=True)
            return embeddings.tolist()
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """Embed documents (used by some ChromaDB versions)."""
            return self(texts)
        
        def embed_query(self, input: str = None, query: str = None, **kwargs) -> List[List[float]]:
            """Embed a single query string. Accepts both 'input' and 'query' kwargs.
            Returns a list of embeddings (even though it's just one) to match ChromaDB's expected format.
            """
            query_text = input or query
            if query_text is None:
                raise ValueError("Either 'input' or 'query' must be provided")
            
            # Ensure query_text is a string, not a list
            if isinstance(query_text, list):
                if len(query_text) == 0:
                    raise ValueError("Query list is empty")
                query_text = query_text[0]
            
            # Convert to string if needed
            query_text = str(query_text)
            
            # Encode and return as a list of embeddings (2D structure)
            embedding = self.model.encode([query_text], convert_to_numpy=True)
            return embedding.tolist()  # Returns List[List[float]]
        
        def name(self) -> str:
            """Return the name of the embedding function."""
            return f"local-sentence-transformer-{self._model_path.name}"
    
    embedding_function = LocalEmbeddingFunction(model_path)

    _collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"description": "Metis RAG knowledge base"}
    )
    return _collection


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks


def index_folder(folder_path: str | Path, clear_existing: bool = False) -> int:
    """
    Index all text files in a folder into ChromaDB.
    Only indexes files that have changed (based on file size).
    
    Returns the number of chunks indexed.
    """
    config = get_config()
    rag_cfg = config.get("rag", {})
    chunk_size = rag_cfg.get("chunk_size", 500)
    chunk_overlap = rag_cfg.get("chunk_overlap", 50)

    collection = get_collection()
    
    # Load existing file metadata
    file_metadata = load_file_metadata()
    
    if clear_existing:
        # Clear existing documents from this folder
        try:
            collection.delete(where={"source_folder": str(folder_path)})
            # Clear metadata for this folder
            file_metadata = {k: v for k, v in file_metadata.items() if not k.startswith(str(folder_path))}
        except Exception:
            pass

    folder = Path(folder_path)
    if not folder.exists():
        print(f"Warning: Folder does not exist: {folder}")
        return 0

    # Resolve folder relative to project root if not absolute
    if not folder.is_absolute():
        folder = (Path(__file__).resolve().parents[1] / folder).resolve()

    supported_extensions = {".txt", ".md", ".py", ".js", ".ts", ".tsx", ".json", ".yaml", ".yml", ".rst"}
    
    documents = []
    metadatas = []
    ids = []
    updated_files = 0
    skipped_files = 0

    for file_path in folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            file_path_str = str(file_path)
            
            # Check if file has changed
            if not clear_existing and not has_file_changed(file_path, file_metadata):
                skipped_files += 1
                continue
            
            # File has changed or is new, delete old chunks for this specific file
            try:
                relative_path = str(file_path.relative_to(folder.parent))
                # Delete all chunks with this source_file
                existing_items = collection.get(where={"source_file": relative_path})
                if existing_items and existing_items.get("ids"):
                    collection.delete(ids=existing_items["ids"])
                    if file_path_str in file_metadata:
                        print(f"Reindexing changed file: {file_path.name} (removed {len(existing_items['ids'])} old chunks)")
            except Exception as e:
                print(f"Error deleting old chunks for {file_path.name}: {e}")
            
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                chunks = chunk_text(content, chunk_size, chunk_overlap)
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    relative_path = str(file_path.relative_to(folder.parent))
                    metadatas.append({
                        "source_file": relative_path,
                        "source_folder": str(folder_path),
                        "chunk_index": i,
                        "file_type": file_path.suffix
                    })
                    # Use consistent ID based on file path and chunk index
                    # This ensures same file always gets same IDs for same chunks
                    file_id = relative_path.replace("\\", "/").replace("/", "_").replace(".", "_")
                    ids.append(f"{file_id}_chunk_{i}")
                
                # Update file metadata
                file_metadata[file_path_str] = get_file_info(file_path)
                updated_files += 1
                
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
                continue

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    # Save updated metadata
    save_file_metadata(file_metadata)
    
    print(f"Indexed {updated_files} files ({len(documents)} chunks), skipped {skipped_files} unchanged files")
    
    return len(documents)


def index_all_folders(clear_existing: bool = False) -> Dict[str, int]:
    """
    Index all folders specified in config. Returns dict of folder -> chunk count.
    Also cleans up deleted files from the index.
    """
    config = get_config()
    rag_cfg = config.get("rag", {})
    folders = rag_cfg.get("folders_to_index", [])
    
    # Load file metadata to check for deleted files
    if not clear_existing:
        file_metadata = load_file_metadata()
        collection = get_collection()
        
        # Check for deleted files and remove their chunks
        deleted_files = []
        for file_path_str in list(file_metadata.keys()):
            file_path = Path(file_path_str)
            if not file_path.exists():
                deleted_files.append(file_path_str)
                # Try to delete chunks for this file
                try:
                    # Get the folder this file belongs to
                    for folder in folders:
                        folder_path = Path(folder)
                        if not folder_path.is_absolute():
                            folder_path = (Path(__file__).resolve().parents[1] / folder).resolve()
                        
                        if file_path.is_relative_to(folder_path.parent):
                            relative_path = str(file_path.relative_to(folder_path.parent))
                            collection.delete(where={"source_file": relative_path})
                            print(f"Removed deleted file from index: {file_path.name}")
                            break
                except Exception as e:
                    print(f"Error removing deleted file {file_path.name}: {e}")
        
        # Remove deleted files from metadata
        if deleted_files:
            for file_path_str in deleted_files:
                del file_metadata[file_path_str]
            save_file_metadata(file_metadata)
            print(f"Cleaned up {len(deleted_files)} deleted files from index")
    
    results = {}
    for folder in folders:
        count = index_folder(folder, clear_existing=clear_existing)
        results[folder] = count
        print(f"Indexed {count} chunks from {folder}")
    
    return results


def retrieve_context(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant context chunks for a query.
    
    Returns list of dicts with 'text', 'metadata', and 'distance'.
    """
    config = get_config()
    rag_cfg = config.get("rag", {})
    
    if not rag_cfg.get("enabled", True):
        return []
    
    if top_k is None:
        top_k = rag_cfg.get("top_k", 3)
    
    collection = get_collection()
    
    # Check if collection has any documents
    try:
        count = collection.count()
        if count == 0:
            print("Warning: RAG collection is empty. No context to retrieve.")
            return []
    except Exception as e:
        print(f"Error checking collection count: {e}")
        return []
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, count)  # Don't request more than available
        )
        
        contexts = []
        # Safely extract results with multiple checks
        if not results:
            return []
        
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])
        
        # Check if we have documents in the first list
        if not documents or len(documents) == 0 or not documents[0]:
            return []
        
        # Iterate through documents
        for i, doc in enumerate(documents[0]):
            context = {"text": doc}
            
            # Safely get metadata
            if metadatas and len(metadatas) > 0 and len(metadatas[0]) > i:
                context["metadata"] = metadatas[0][i]
            else:
                context["metadata"] = {}
            
            # Safely get distance
            if distances and len(distances) > 0 and len(distances[0]) > i:
                context["distance"] = distances[0][i]
            else:
                context["distance"] = 0.0
            
            contexts.append(context)
        
        return contexts
    except Exception as e:
        import traceback
        print(f"Error retrieving context: {e}")
        print(traceback.format_exc())
        return []


def format_context_for_prompt(contexts: List[Dict[str, Any]]) -> str:
    """Format retrieved contexts into a string suitable for injection into the prompt."""
    if not contexts:
        return ""
    
    formatted = "Relevant context from knowledge base:\n\n"
    for i, ctx in enumerate(contexts, 1):
        source = ctx["metadata"].get("source_file", "unknown")
        formatted += f"[{i}] Source: {source}\n{ctx['text']}\n\n"
    
    return formatted.strip()
