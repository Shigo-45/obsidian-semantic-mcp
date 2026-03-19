"""ChromaDB wrapper for persistent local vector storage."""

import json

import chromadb

from config import CHROMA_PERSIST_DIR, EMBEDDING_DIM

# Lazy-initialized ChromaDB client and collection
_client = None
_collection = None


def _get_collection():
    """Return the ChromaDB collection, initializing on first access."""
    global _client, _collection
    if _collection is None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        _collection = _client.get_or_create_collection(
            name="vault",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def upsert_chunk(
    file_path: str,
    chunk_id: str,
    embedding: list[float],
    chunk_text: str,
    file_type: str,
    extra_metadata: dict = None,
) -> None:
    """Upsert a single chunk with its embedding and metadata.

    Args:
        file_path: Vault-relative (or absolute) path of the source file.
        chunk_id: Unique chunk identifier, e.g. ``"path/to/file.md::chunk_0"``.
        embedding: Pre-computed embedding vector of length EMBEDDING_DIM.
        chunk_text: Raw text of the chunk (stored as the ChromaDB document).
        file_type: One of the keys in SUPPORTED_EXTENSIONS («text», «pdf», etc.).
        extra_metadata: Optional additional key/value pairs merged into metadata.
    """
    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: got {len(embedding)}, expected {EMBEDDING_DIM}"
        )

    metadata = {
        "file_path": file_path,
        "chunk_id": chunk_id,
        "file_type": file_type,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    # ChromaDB metadata values must be str, int, float, or bool
    for key, value in metadata.items():
        if isinstance(value, (list, dict)):
            metadata[key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            metadata[key] = ""

    _get_collection().upsert(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[chunk_text],
        metadatas=[metadata],
    )


def query(embedding: list[float], n_results: int = 10, file_type: str | None = None) -> list[dict]:
    """Return the top-k most similar chunks.

    Args:
        embedding: Query embedding vector.
        n_results: Number of results to return.
        file_type: Optional filter — only return chunks of this type
                   (e.g. "text", "pdf", "image", "audio", "video").

    Returns:
        List of dicts, each containing:
            - ``file_path``: source file path
            - ``chunk_id``: chunk identifier
            - ``file_type``: type string
            - ``score``: cosine similarity (0-1, higher = more similar)
            - ``snippet``: first 300 characters of chunk text
    """
    total = _get_collection().count()
    # Clamp n_results to the number of indexed chunks to avoid ChromaDB errors
    k = min(n_results, total) if total > 0 else 0
    if k == 0:
        return []

    query_kwargs: dict = {
        "query_embeddings": [embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if file_type:
        query_kwargs["where"] = {"file_type": file_type}

    results = _get_collection().query(**query_kwargs)

    output = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
        output.append(
            {
                "file_path": meta.get("file_path", ""),
                "chunk_id": meta.get("chunk_id", doc_id),
                "file_type": meta.get("file_type", ""),
                "score": round(1.0 - dist, 4),
                "snippet": doc_text[:300] if doc_text else "",
            }
        )

    return output


def get_status() -> dict:
    """Return index statistics.

    Returns:
        Dict with:
            - ``total_chunks``: total number of indexed chunks
            - ``by_file_type``: mapping of file_type -> chunk count
            - ``collection_name``: name of the ChromaDB collection
    """
    col = _get_collection()
    total = col.count()
    by_file_type: dict[str, int] = {}

    if total > 0:
        # Fetch all metadatas to aggregate by file_type
        all_meta = col.get(include=["metadatas"])["metadatas"]
        for meta in all_meta:
            ft = meta.get("file_type", "unknown")
            by_file_type[ft] = by_file_type.get(ft, 0) + 1

    return {
        "total_chunks": total,
        "by_file_type": by_file_type,
        "collection_name": col.name,
    }


def delete_chunks_for_file(file_path: str) -> None:
    """Delete all chunks belonging to a specific file.

    Useful when re-indexing a file to remove stale vectors before upserting
    the new ones.

    Args:
        file_path: The file path stored in chunk metadata.
    """
    results = _get_collection().get(
        where={"file_path": file_path},
        include=[],  # only need IDs
    )
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        _get_collection().delete(ids=ids_to_delete)


def get_indexed_files() -> list[str]:
    """Return a deduplicated list of all file paths present in the index.

    Returns:
        Sorted list of unique file path strings.
    """
    col = _get_collection()
    total = col.count()
    if total == 0:
        return []

    all_meta = col.get(include=["metadatas"])["metadatas"]
    paths = {meta.get("file_path", "") for meta in all_meta if meta.get("file_path")}
    return sorted(paths)
