"""ChromaDB wrapper for persistent local vector storage."""

import json
import threading

import chromadb

from config import CHROMA_PERSIST_DIR, EMBEDDING_DIM

# Lazy-initialized ChromaDB client and collection
_client = None
_collection = None
_init_lock = threading.Lock()
# Serializes all write operations (upsert/delete) — ChromaDB's PersistentClient
# uses a Rust-backed SQLite and is not safe for concurrent writes.
_write_lock = threading.Lock()


def _get_collection():
    """Return the ChromaDB collection, initializing on first access."""
    global _client, _collection
    if _collection is None:
        with _init_lock:
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

    with _write_lock:
        _get_collection().upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk_text],
            metadatas=[metadata],
        )


def query(embedding: list[float], n_results: int = 10, file_type: str | None = None, snippet_length: int = 300) -> list[dict]:
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
                "snippet": doc_text[:snippet_length] if doc_text else "",
            }
        )

    return output


# Batch size for paginated metadata fetches — keeps each ChromaDB get() call well
# under SQLite's SQLITE_MAX_VARIABLE_NUMBER (default 32,766 for ChromaDB's build).
# Using limit/offset avoids the ``WHERE id IN (...)`` explosion that occurs when
# calling get() without filters on large collections.
_GET_BATCH_SIZE = 2000


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
        # Fetch metadatas in batches to avoid SQLite variable-limit errors on
        # large collections.  Using limit/offset causes ChromaDB to emit
        # ``LIMIT N OFFSET M`` instead of ``WHERE id IN (id1, id2, ...)``.
        offset = 0
        while True:
            batch = col.get(
                limit=_GET_BATCH_SIZE,
                offset=offset,
                include=["metadatas"],
            )
            metas = batch["metadatas"]
            if not metas:
                break
            for meta in metas:
                ft = meta.get("file_type", "unknown")
                by_file_type[ft] = by_file_type.get(ft, 0) + 1
            offset += len(metas)
            if len(metas) < _GET_BATCH_SIZE:
                break

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
    with _write_lock:
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

    paths: set[str] = set()
    offset = 0
    while True:
        batch = col.get(
            limit=_GET_BATCH_SIZE,
            offset=offset,
            include=["metadatas"],
        )
        metas = batch["metadatas"]
        if not metas:
            break
        for meta in metas:
            fp = meta.get("file_path", "")
            if fp:
                paths.add(fp)
        offset += len(metas)
        if len(metas) < _GET_BATCH_SIZE:
            break

    return sorted(paths)


# ---------------------------------------------------------------------------
# Lexical / hybrid search helpers
# ---------------------------------------------------------------------------


def _case_variants(term: str) -> list[str]:
    """Return the term plus case variants that might match stored text.

    ChromaDB ``$contains`` is case-sensitive.  A query for ``mt-keima`` will
    not match ``mt-Keima`` stored in a chunk.  We probe the original form plus
    lowercased plus a few common capitalisations so the user doesn't have to
    guess the exact capitalisation used in their notes.

    Returns a de-duplicated list (original always first so the exact match
    gets priority when present).
    """
    variants = [term]
    lower = term.lower()
    if lower not in variants:
        variants.append(lower)
    upper = term.upper()
    if upper not in variants:
        variants.append(upper)
    title = term.title()
    if title not in variants:
        variants.append(title)
    return variants


def lexical_query(
    terms: list[str],
    n_results: int = 10,
    file_type: str | None = None,
    snippet_length: int = 300,
) -> list[dict]:
    """Search chunks by exact/substring term matching (no embeddings).

    For each term, queries ChromaDB with ``where_document $contains``,
    probing case variants to handle capitalisation differences (e.g.
    ``mt-keima`` will match ``mt-Keima`` in stored text).

    Results are scored by how many distinct query terms appear in each
    chunk, then by match order within the term.  Chunks matching more
    terms rank higher.

    Args:
        terms: List of search terms (whitespace-split query tokens).
        n_results: Maximum results to return.
        file_type: Optional filter by file type.
        snippet_length: Characters per snippet.

    Returns:
        Same structure as :func:`query`.
    """
    if not terms:
        return []

    col = _get_collection()
    total = col.count()
    if total == 0:
        return []

    # Per-chunk-id → (accumulated_score, doc_text, metadata, set[term_idx])
    hits: dict[str, tuple[float, str, dict, set[int]]] = {}

    for term_idx, term in enumerate(terms):
        for variant in _case_variants(term):
            try:
                results = col.get(
                    where_document={"$contains": variant},
                    include=["documents", "metadatas"],
                )
            except Exception:
                # Some ChromaDB versions reject unknown operators or other
                # edge cases — skip this variant gracefully.
                continue

            ids = results.get("ids", [])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            for doc_id, doc_text, meta in zip(ids, documents, metadatas):
                if file_type and meta.get("file_type") != file_type:
                    continue
                # Score: earlier terms rank higher.  First variant that
                # matches contributes the term score; subsequent variants
                # of the same term do NOT inflate match_terms.
                term_score = 1.0 / (1.0 + term_idx)  # 1.0, 0.5, 0.33, ...
                existing = hits.get(doc_id)
                if existing is None:
                    hits[doc_id] = (term_score, doc_text, meta, {term_idx})
                elif term_idx not in existing[3]:
                    prev_score, prev_doc, prev_meta, prev_set = existing
                    hits[doc_id] = (
                        prev_score + term_score,
                        prev_doc,
                        prev_meta,
                        prev_set | {term_idx},
                    )
                # else: term_idx already matched via another variant — skip

    if not hits:
        return []

    # Sort by (num_matched_terms DESC, accumulated_score DESC)
    sorted_hits = sorted(
        hits.items(),
        key=lambda kv: (len(kv[1][3]), kv[1][0]),
        reverse=True,
    )

    output = []
    for doc_id, (acc_score, doc_text, meta, term_set) in sorted_hits[:n_results]:
        match_count = len(term_set)
        # Normalise score: base 0.80 + 0.05 per extra matched term, capped at 0.99
        lexical_score = min(0.99, 0.80 + 0.05 * (match_count - 1))
        output.append(
            {
                "file_path": meta.get("file_path", ""),
                "chunk_id": meta.get("chunk_id", doc_id),
                "file_type": meta.get("file_type", ""),
                "score": round(lexical_score, 4),
                "snippet": doc_text[:snippet_length] if doc_text else "",
                "match_terms": match_count,
            }
        )

    return output


# ---------------------------------------------------------------------------
# Stop-word set — common English words that are too generic for lexical search.
# ---------------------------------------------------------------------------
_LEXICAL_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "not", "only", "own", "same", "so", "than", "too",
    "very", "and", "but", "or", "nor", "if", "while", "about", "up",
    "out", "it", "its", "this", "that", "these", "those", "just", "also",
    "now", "well", "way", "even", "new", "want", "because", "any", "every",
    "which", "what", "who", "whom", "whose",
})


def _extract_lexical_terms(query: str) -> list[str]:
    """Extract search-worthy terms from a query string.

    Splits on whitespace, drops single-character non-CJK words and
    common English stop-words.  Preserves hyphenated terms (mt-Keima),
    Chinese characters, and mixed-language tokens.
    """
    raw_terms = query.split()
    terms: list[str] = []
    for t in raw_terms:
        t = t.strip().rstrip(".,;:!?")
        if not t:
            continue
        # Keep any term that contains CJK characters
        has_cjk = any("\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" for c in t)
        if has_cjk:
            terms.append(t)
        elif len(t) >= 2 and t.lower() not in _LEXICAL_STOP_WORDS:
            terms.append(t)
    return terms
