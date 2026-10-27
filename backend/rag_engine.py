from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from backend.config import PROJECT_ROOT, get_config, resolve_path
from backend.llama_engine import chat_completion

# Bump when the way documents/queries are embedded changes; an index built with an
# older scheme is rebuilt automatically on startup.
EMBEDDING_SCHEME = "st-query-document-prompts-v1"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".js", ".ts", ".tsx", ".json", ".yaml", ".yml", ".rst"}

_chroma_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None
_reranker_model = None
_reranker_loaded = False
_init_lock = threading.RLock()
# Serialises indexing runs (startup scan, manual reindex, memory archiving).
index_lock = threading.Lock()


def _rag_cfg() -> Dict[str, Any]:
    return get_config().get("rag", {})


# ---------------------------------------------------------------------------
# File change tracking
# ---------------------------------------------------------------------------
def get_file_metadata_path() -> Path:
    persist_path = resolve_path(_rag_cfg().get("persist_directory", ".chromadb"))
    persist_path.mkdir(parents=True, exist_ok=True)
    return persist_path / "file_metadata.json"


def load_file_metadata() -> Dict[str, Dict[str, Any]]:
    path = get_file_metadata_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file metadata: {e}")
        return {}


def save_file_metadata(metadata: Dict[str, Dict[str, Any]]) -> None:
    path = get_file_metadata_path()
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        tmp.replace(path)
    except Exception as e:
        print(f"Error saving file metadata: {e}")


def get_file_info(file_path: Path) -> Dict[str, Any]:
    try:
        stat = file_path.stat()
        return {"size": stat.st_size, "mtime": stat.st_mtime}
    except OSError:
        return {"size": 0, "mtime": 0}


def has_file_changed(file_path: Path, stored_metadata: Dict[str, Any]) -> bool:
    stored = stored_metadata.get(str(file_path))
    if not stored:
        return True
    current = get_file_info(file_path)
    return current["size"] != stored.get("size") or current["mtime"] != stored.get("mtime", current["mtime"])


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
class LocalEmbeddingFunction:
    """Embeds with a local SentenceTransformer, using the model's query/document prompts when it has them."""

    def __init__(self, model_path: Path):
        import torch
        from sentence_transformers import SentenceTransformer

        # rag.embedding_device: "cpu" frees ~1 GB of VRAM (weights + torch's CUDA context) for games, at the cost
        # of slower embedding (a few ms -> tens of ms per message; reading web pages takes longer).
        pref = str(get_config().get("rag", {}).get("embedding_device", "auto")).lower()
        device = "cuda" if pref in ("cuda", "auto") and torch.cuda.is_available() else "cpu"
        # bfloat16 halves VRAM (~1.2 GB -> ~0.6 GB for EmbeddingGemma) with embeddings that are
        # identical in practice (cosine 0.9999 vs fp32). fp16 is not used: Gemma activations overflow.
        precision = str(get_config().get("rag", {}).get("embedding_precision", "auto")).lower()
        kwargs = {}
        if device == "cuda" and precision != "fp32" and torch.cuda.is_bf16_supported():
            kwargs["model_kwargs"] = {"dtype": torch.bfloat16}
        self.model = SentenceTransformer(str(model_path), device=device, **kwargs)
        self._dtype = "bf16" if kwargs else "fp32"
        self._model_path = model_path
        prompts = getattr(self.model, "prompts", {}) or {}
        self._has_prompts = "query" in prompts and "document" in prompts
        print(f"✓ Embedding model loaded on {device} ({self._dtype})" + (" with query/document prompts" if self._has_prompts else ""))

    def __call__(self, input: List[str]) -> List[List[float]]:
        if self._has_prompts:
            return self.model.encode_document(list(input), convert_to_numpy=True).tolist()
        return self.model.encode(list(input), convert_to_numpy=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self(texts)

    def embed_query(self, input: Any = None, query: Any = None, **kwargs) -> List[List[float]]:
        texts = input if input is not None else query
        if texts is None:
            raise ValueError("Either 'input' or 'query' must be provided")
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) for t in texts]
        if self._has_prompts:
            return self.model.encode_query(texts, convert_to_numpy=True).tolist()
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def name(self) -> str:
        return f"local-sentence-transformer-{self._model_path.name}"


def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    with _init_lock:
        if _chroma_client is None:
            persist_path = resolve_path(_rag_cfg().get("persist_directory", ".chromadb"))
            persist_path.mkdir(parents=True, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(
                path=str(persist_path), settings=Settings(anonymized_telemetry=False)
            )
        return _chroma_client


def get_collection() -> chromadb.Collection:
    global _collection
    with _init_lock:
        if _collection is not None:
            return _collection

        rag_cfg = _rag_cfg()
        name = rag_cfg.get("collection_name", "metis_knowledge")
        model_path = resolve_path(rag_cfg.get("embedding_model", "rag-models/embeddinggemma-300m"))
        if not model_path.exists():
            raise FileNotFoundError(f"Embedding model not found at: {model_path}")

        embedding_function = LocalEmbeddingFunction(model_path)
        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function,
            metadata={"description": "Metis RAG knowledge base", "embedding_scheme": EMBEDDING_SCHEME},
        )
        if (collection.metadata or {}).get("embedding_scheme") != EMBEDDING_SCHEME:
            print("RAG index was built with an older embedding scheme; rebuilding it.")
            client.delete_collection(name)
            save_file_metadata({})
            collection = client.create_collection(
                name=name,
                embedding_function=embedding_function,
                metadata={"description": "Metis RAG knowledge base", "embedding_scheme": EMBEDDING_SCHEME},
            )
        _collection = collection
        return _collection


def get_embedder() -> LocalEmbeddingFunction:
    """The knowledge base's embedding model (also used to pick the relevant parts of web pages)."""
    return get_collection()._embedding_function


_embedder_failed_at = 0.0


def embed_texts(texts: List[str], query: bool = False) -> Optional[List[List[float]]]:
    """
    Vectors for `texts` (as search queries, or as documents) from the knowledge base's model, or None when it
    isn't available (knowledge base off, model missing): the memory system then matches by words. After a
    failure it doesn't retry for a minute, so every reply doesn't wait on a model that isn't there.
    """
    global _embedder_failed_at
    if not _rag_cfg().get("enabled", False) or not texts or time.time() - _embedder_failed_at < 60:
        return None
    try:
        embedder = get_embedder()
        return embedder.embed_query(list(texts)) if query else embedder.embed_documents(list(texts))
    except Exception as e:
        _embedder_failed_at = time.time()
        print(f"Warning: embedding model unavailable ({e}); memories are matched by words")
        return None


def clear_collection() -> int:
    """Delete every chunk and forget all file metadata. Returns the number of chunks removed."""
    collection = get_collection()
    with index_lock:
        count = collection.count()
        ids = collection.get(include=[]).get("ids") or []
        for batch in _batched(ids, 1000):
            collection.delete(ids=batch)
        save_file_metadata({})
    return count


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
def get_reranker_model():
    global _reranker_model, _reranker_loaded
    with _init_lock:
        if _reranker_loaded:
            return _reranker_model
        _reranker_loaded = True

        rag_cfg = _rag_cfg()
        if not rag_cfg.get("use_reranker", False):
            return None
        model_path = resolve_path(rag_cfg.get("reranker_model", "rag-models/bge-reranker-base"))
        if not model_path.exists():
            print(f"Warning: reranker model not found at {model_path}; reranking disabled")
            return None
        try:
            import torch
            from sentence_transformers import CrossEncoder

            pref = str(rag_cfg.get("reranker_device", "cpu")).lower()
            device = "cuda" if pref in ("cuda", "auto") and torch.cuda.is_available() else "cpu"
            # On GPU, fp16 is ~9x faster than CPU (9 ms vs 86 ms for 6 chunks) at ~560 MB of VRAM.
            kwargs = {"model_kwargs": {"dtype": torch.float16}} if device == "cuda" else {}
            _reranker_model = CrossEncoder(str(model_path), device=device, **kwargs)
            print(f"✓ Reranker loaded on {device}")
        except Exception as e:
            print(f"Error loading reranker model: {e}")
            _reranker_model = None
        return _reranker_model


def rerank_contexts(query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score contexts with the cross-encoder and sort best-first (adds 'rerank_score')."""
    reranker = get_reranker_model()
    if reranker is None or not contexts:
        return contexts
    try:
        scores = reranker.predict([[query, c["text"]] for c in contexts])
        for ctx, score in zip(contexts, scores):
            ctx["rerank_score"] = float(score)
        return sorted(contexts, key=lambda c: c["rerank_score"], reverse=True)
    except Exception as e:
        print(f"Error during reranking: {e}")
        return contexts


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks of roughly `chunk_size` tokens."""
    step = max(1, chunk_size - overlap)
    try:
        from backend.token_utils import decode_tokens, encode_text

        tokens = encode_text(text)
        chunks = []
        for start in range(0, len(tokens), step):
            chunk = decode_tokens(tokens[start:start + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            if start + chunk_size >= len(tokens):
                break
        return chunks
    except Exception as e:
        print(f"Tokenizer unavailable ({e}); falling back to character chunking")
        size, step = chunk_size * 4, step * 4
        return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def _batched(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _chunk_id(relative_path: str, index: int) -> str:
    return f"{relative_path.replace(chr(92), '/')}::chunk_{index}"


def _relative_source(file_path: Path, folder: Path) -> str:
    return str(file_path.relative_to(folder.parent))


def index_folder(folder_path: str | Path, clear_existing: bool = False) -> int:
    """
    Index text files in a folder. Unchanged files (same size and mtime) are skipped.
    Returns the number of chunks written.
    """
    rag_cfg = _rag_cfg()
    chunk_size = int(rag_cfg.get("chunk_size", 750))
    chunk_overlap = int(rag_cfg.get("chunk_overlap", 75))
    folder = resolve_path(folder_path)
    if not folder.exists():
        print(f"Warning: folder does not exist: {folder}")
        return 0

    collection = get_collection()
    with index_lock:
        file_metadata = load_file_metadata()
        if clear_existing:
            collection.delete(where={"source_folder": str(folder_path)})
            prefix = str(folder)
            file_metadata = {k: v for k, v in file_metadata.items() if not k.startswith(prefix)}

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        updated = skipped = 0

        for file_path in folder.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if not clear_existing and not has_file_changed(file_path, file_metadata):
                skipped += 1
                continue

            relative = _relative_source(file_path, folder)
            try:
                collection.delete(where={"source_file": relative})
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            for i, chunk in enumerate(chunk_text(content, chunk_size, chunk_overlap)):
                documents.append(chunk)
                metadatas.append({
                    "source_file": relative,
                    "source_folder": str(folder_path),
                    "chunk_index": i,
                    "file_type": file_path.suffix,
                })
                ids.append(_chunk_id(relative, i))
            file_metadata[str(file_path)] = get_file_info(file_path)
            updated += 1

        for batch in _batched(list(range(len(documents))), 256):
            collection.upsert(
                documents=[documents[i] for i in batch],
                metadatas=[metadatas[i] for i in batch],
                ids=[ids[i] for i in batch],
            )
        save_file_metadata(file_metadata)

    print(f"Indexed {updated} files ({len(documents)} chunks) from {folder}, skipped {skipped} unchanged")
    return len(documents)


def _remove_deleted_files(folders: List[str]) -> None:
    collection = get_collection()
    with index_lock:
        file_metadata = load_file_metadata()
        deleted = [p for p in file_metadata if not Path(p).exists()]
        for path_str in deleted:
            file_path = Path(path_str)
            for folder in folders:
                folder_path = resolve_path(folder)
                if file_path.is_relative_to(folder_path):
                    collection.delete(where={"source_file": _relative_source(file_path, folder_path)})
                    break
            del file_metadata[path_str]
        if deleted:
            save_file_metadata(file_metadata)
            print(f"Removed {len(deleted)} deleted file(s) from the index")


def index_all_folders(clear_existing: bool = False) -> Dict[str, int]:
    """Index every folder from config. Also drops chunks of files that no longer exist."""
    folders = [str(f) for f in _rag_cfg().get("folders_to_index", [])]
    if not clear_existing:
        _remove_deleted_files(folders)
    return {folder: index_folder(folder, clear_existing=clear_existing) for folder in folders}


def reindex_path_if_tracked(path: Path) -> None:
    """Incrementally re-index the configured folder containing `path` (used after memory archiving)."""
    if not _rag_cfg().get("enabled", False):
        return
    path = path.resolve()
    for folder in _rag_cfg().get("folders_to_index", []):
        folder_path = resolve_path(folder)
        if path == folder_path or path.is_relative_to(folder_path):
            index_folder(folder)
            return


# ---------------------------------------------------------------------------
# Query generation (knowledge_search and remember tools)
# ---------------------------------------------------------------------------
def _parse_query_lines(text: str, limit: int) -> List[str]:
    queries = []
    for line in text.splitlines():
        q = re.sub(r"^\s*(?:\d+\s*[-.)]+|[-*•])\s*", "", line).strip().strip('"').strip("'")
        if q and len(q) <= 200 and "RULES:" not in q:
            queries.append(q)
        if len(queries) >= limit:
            break
    return queries


# A small model follows worked examples far better than a list of rules (the old prompt mostly got
# the message echoed back). Memories about people are stored as "user personal information: <name> ...".
_KNOWLEDGE_QUERY_SYSTEM = (
    "You write search queries for Metis's memory: notes and what she remembers about people. Given a chat, "
    "write short keyword queries that would find what helps her reply to the latest message. Use people's "
    "names, and take the topic from earlier messages when the latest one says 'it' or 'that'. For things about "
    "a person, start with 'user personal information' and their name. One query per line, nothing else."
)
_KNOWLEDGE_QUERY_EXAMPLES = [
    ("Earlier messages:\n[Robin]: I've started learning guitar\n\nLatest message: [Robin]: what song should I try next?\n\n"
     "Write 2 queries.",
     "user personal information Robin guitar\nbeginner guitar songs"),
    ("Latest message: [Sava]: do you remember what my cat is called?\n\nWrite 1 query.",
     "user personal information Sava cat name"),
    ("Earlier messages:\n[Ana]: my exam is on friday\n\nLatest message: [Ana]: I'm so nervous about it\n\n"
     "Write 2 queries.",
     "user personal information Ana exam\nexam nerves"),
]


def generate_optimized_queries(
    last_user_message: str,
    conversation_context: str = "",
    num_queries: int = 3,
    voice_mode: bool = False,
) -> List[str]:
    """Queries for the knowledge_search tool (context: earlier messages, one per line)."""
    examples: List[Dict[str, str]] = []
    for question, answer in _KNOWLEDGE_QUERY_EXAMPLES:
        examples += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    prompt = ((f"Earlier messages:\n{conversation_context}\n\n" if conversation_context else "")
              + f"Latest message: {last_user_message}\n\nWrite {num_queries} {'query' if num_queries == 1 else 'queries'}.")
    try:
        response = chat_completion(
            [{"role": "system", "content": _KNOWLEDGE_QUERY_SYSTEM}, *examples, {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=24 * num_queries,
        )
        queries = _parse_query_lines(response, num_queries)
    except Exception as e:
        print(f"Query generation error: {e}")
        queries = []
    return queries or [last_user_message]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _empty_result() -> Dict[str, List[Dict[str, Any]]]:
    return {"accepted": [], "rejected_by_distance": [], "rejected_by_score": []}


def retrieve_context(query: str, top_k: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve chunks for one query.

    Pipeline: vector search (top_k) -> distance filter -> rerank -> score filter
    -> keep `reranker_top_k`. Always returns {"accepted", "rejected_by_distance", "rejected_by_score"}.
    """
    rag_cfg = _rag_cfg()
    if not rag_cfg.get("enabled", True) or not query.strip():
        return _empty_result()
    top_k = int(top_k or rag_cfg.get("top_k", 3))

    try:
        collection = get_collection()
        count = collection.count()
        if count == 0:
            return _empty_result()
        results = collection.query(query_texts=[query], n_results=min(top_k, count))
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return _empty_result()

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    contexts = [
        {
            "text": doc,
            "metadata": metadatas[i] if i < len(metadatas) else {},
            "distance": distances[i] if i < len(distances) else 0.0,
        }
        for i, doc in enumerate(documents)
    ]

    max_distance = rag_cfg.get("max_distance", -1)
    rejected_by_distance: List[Dict[str, Any]] = []
    candidates = contexts
    if max_distance != -1:
        candidates = [c for c in contexts if c["distance"] <= max_distance]
        rejected_by_distance = [dict(c, rejection_reason="distance") for c in contexts if c["distance"] > max_distance]

    rejected_by_score: List[Dict[str, Any]] = []
    keep = top_k
    if rag_cfg.get("use_reranker", False) and candidates:
        candidates = rerank_contexts(query, candidates)
        min_score = rag_cfg.get("reranker_min_score", -1)
        if min_score != -1:
            rejected_by_score = [
                dict(c, rejection_reason="score") for c in candidates
                if c.get("rerank_score") is not None and c["rerank_score"] < float(min_score)
            ]
            candidates = [c for c in candidates if c.get("rerank_score") is None or c["rerank_score"] >= float(min_score)]
        keep = int(rag_cfg.get("reranker_top_k", top_k))

    return {
        "accepted": candidates[:keep],
        "rejected_by_distance": rejected_by_distance,
        "rejected_by_score": rejected_by_score + [dict(c, rejection_reason="rank") for c in candidates[keep:]],
    }


def quick_search(query: str, max_distance: float = 1.2, limit: int = 2) -> Dict[str, Any]:
    """
    One vector lookup, no reranker and no generated queries: ~50 ms instead of 1-2 s, for every message.
    Only close matches count. On the notes index, chunks that answer a message were at distance
    0.94-1.14 and small talk never came closer than 1.41, so a distance cutoff alone tells them apart.
    The full search (generated queries + reranker) is the `remember` tool, used when the model asks for it.
    """
    if not _rag_cfg().get("enabled", True) or not query.strip():
        return {"queries": [query], "accepted": [], "rejected": []}
    try:
        collection = get_collection()
        count = collection.count()
        results = collection.query(query_texts=[query], n_results=min(limit + 2, count)) if count else {}
    except Exception as e:
        print(f"Error retrieving context: {e}")
        results = {}
    contexts = [
        {"text": doc, "metadata": md or {}, "distance": dist}
        for doc, md, dist in zip((results.get("documents") or [[]])[0], (results.get("metadatas") or [[]])[0],
                                 (results.get("distances") or [[]])[0])
    ]
    close = [c for c in contexts if c["distance"] <= max_distance]
    return {
        "queries": [query],
        "accepted": close[:limit],
        "rejected": [dict(c, rejection_reason="distance") for c in contexts if c["distance"] > max_distance]
        + [dict(c, rejection_reason="rank") for c in close[limit:]],
    }


def _chunk_key(ctx: Dict[str, Any]) -> Tuple[Any, Any]:
    md = ctx.get("metadata") or {}
    if md.get("source_file") is None or md.get("chunk_index") is None:
        return (None, ctx.get("text", "")[:160])
    return (md["source_file"], md["chunk_index"])


def search(queries: List[str], top_k: Optional[int] = None) -> Dict[str, Any]:
    """
    Run several queries and merge the results, de-duplicating chunks.
    A chunk accepted by any query counts as accepted.
    """
    accepted: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    rejected: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for query in dict.fromkeys(q for q in queries if q and q.strip()):
        result = retrieve_context(query, top_k=top_k)
        for ctx in result["accepted"]:
            key = _chunk_key(ctx)
            best = accepted.get(key)
            if best is None or (ctx.get("rerank_score") or -1e9) > (best.get("rerank_score") or -1e9):
                accepted[key] = ctx
            rejected.pop(key, None)
        for ctx in result["rejected_by_distance"] + result["rejected_by_score"]:
            key = _chunk_key(ctx)
            if key not in accepted:
                rejected.setdefault(key, ctx)

    def sort_key(c: Dict[str, Any]) -> Tuple[float, float]:
        score = c.get("rerank_score")
        return (-(score if score is not None else -1e9), c.get("distance", 0.0))

    return {
        "queries": list(dict.fromkeys(queries)),
        "accepted": sorted(accepted.values(), key=sort_key),
        "rejected": sorted(rejected.values(), key=lambda c: c.get("distance", 0.0)),
    }


def results_for_ui(search_result: Dict[str, Any], original_query: str, **extra: Any) -> Dict[str, Any]:
    """Shape a search() result for the dashboard's knowledge panel."""
    def item(ctx: Dict[str, Any], used: bool) -> Dict[str, Any]:
        md = ctx.get("metadata") or {}
        out = {
            "source_file": md.get("source_file", "unknown"),
            "distance": ctx.get("distance", 0.0),
            "rerank_score": ctx.get("rerank_score"),
            "text_preview": ctx.get("text", "")[:200],
            "chunk_index": md.get("chunk_index", 0),
            "used": used,
        }
        if not used:
            out["rejection_reason"] = ctx.get("rejection_reason", "distance")
        return out

    return {
        "queries": search_result.get("queries", []),
        "original_query": original_query,
        "results": [item(c, True) for c in search_result.get("accepted", [])]
        + [item(c, False) for c in search_result.get("rejected", [])],
        **extra,
    }


def format_context_for_prompt(contexts: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks for injection into the prompt."""
    if not contexts:
        return ""
    parts = []
    for i, ctx in enumerate(contexts, 1):
        source = (ctx.get("metadata") or {}).get("source_file", "unknown")
        parts.append(f"[{i}] Source: {source}\n{ctx['text'].strip()}")
    return "\n\n".join(parts)


__all__ = [
    "PROJECT_ROOT",
    "get_collection",
    "clear_collection",
    "index_all_folders",
    "index_folder",
    "reindex_path_if_tracked",
    "retrieve_context",
    "search",
    "results_for_ui",
    "format_context_for_prompt",
    "generate_optimized_queries",
    "get_file_metadata_path",
]
