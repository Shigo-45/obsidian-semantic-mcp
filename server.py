"""FastMCP server and CLI ingestion for obsidian-semantic-mcp."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import index
import tracker
from config import SUPPORTED_EXTENSIONS, VAULT_PATH
from embedder import embed_audio, embed_image, embed_text, embed_texts
from ingestion.audio import chunk_audio
from ingestion.canvas import chunk_canvas
from ingestion.image import chunk_image
from ingestion.markdown import chunk_markdown
from ingestion.pdf import chunk_pdf
from ingestion.video import chunk_video

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("obsidian-semantic")


_VALID_FILE_TYPES = {"text", "pdf", "image", "audio", "video"}


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def vault_semantic_search(
    query: str,
    n_results: int = 10,
    file_type: str | None = None,
    snippet_length: int = 300,
) -> str:
    """Search the Obsidian vault by meaning. Returns ranked results with file paths,
    relevance scores, and text snippets. Use this to find notes related to a topic
    even if they don't contain the exact search terms.

    Args:
        query: Natural language search query
        n_results: Number of results to return (default 10, max 50)
        file_type: Optional filter — "text" (md/canvas), "pdf", "image", "audio", "video"
        snippet_length: Characters to return per snippet (default 300, max 2000)

    Returns:
        JSON string with search results including file paths, scores, and snippets
    """
    if file_type is not None and file_type not in _VALID_FILE_TYPES:
        return json.dumps(
            {"error": f"Invalid file_type '{file_type}'. Must be one of: {sorted(_VALID_FILE_TYPES)}"},
            ensure_ascii=False,
        )
    n_results = min(max(1, n_results), 50)
    snippet_length = min(max(50, snippet_length), 2000)
    try:
        embedding = embed_text(query, task_type="RETRIEVAL_QUERY")
        results = index.query(embedding, n_results=n_results, file_type=file_type, snippet_length=snippet_length)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def vault_index_status() -> str:
    """Get the current status of the vault index.

    Returns:
        JSON string with total chunks, counts by file type, and collection info
    """
    status = index.get_status()
    status["tracker"] = tracker.get_stats()
    return json.dumps(status, ensure_ascii=False, indent=2)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def vault_reindex(file_path: str) -> str:
    """Re-index a specific file in the vault. Embeds updated content and refreshes
    its entry in the vector index.

    To re-index the entire vault, run the CLI: uv run python server.py ingest --full

    Args:
        file_path: Absolute or vault-relative path to the file to re-index.

    Returns:
        JSON string with chunks_indexed and errors counts
    """
    try:
        fp = Path(file_path)
        # Try resolving relative to vault root if not absolute
        if not fp.is_file() and VAULT_PATH:
            fp = VAULT_PATH / file_path
        if not fp.is_file():
            return json.dumps(
                {"error": f"File not found: {file_path}. Provide an absolute path or a path relative to the vault root."},
                ensure_ascii=False,
            )
        indexed, errors = _ingest_single_file(fp)
        tracker.mark_indexed(fp)
        return json.dumps({"file": str(fp), "chunks_indexed": indexed, "errors": errors}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def vault_get_file(file_path: str) -> str:
    """Retrieve the full content of a file from the vault.

    For text files (.md, .canvas), returns the raw text content.
    For PDFs, returns extracted text separated by page breaks.
    For binary files (image, audio, video), returns file metadata.

    Args:
        file_path: Absolute or vault-relative path to the file

    Returns:
        File content as text, or JSON metadata for binary files
    """
    fp = Path(file_path)
    if not fp.is_file() and VAULT_PATH:
        fp = VAULT_PATH / file_path
    if not fp.is_file():
        return json.dumps(
            {"error": f"File not found: {file_path}. Provide an absolute path or a path relative to the vault root."},
            ensure_ascii=False,
        )

    ext = fp.suffix.lower()
    if ext in _TEXT_EXTS:
        try:
            return fp.read_text(encoding="utf-8")
        except Exception as exc:
            return json.dumps({"error": f"Failed to read file: {exc}"}, ensure_ascii=False)
    elif ext in _PDF_EXTS:
        chunks = chunk_pdf(fp)
        text = "\n\n---\n\n".join(c["text"] for c in chunks)
        return text if text else json.dumps({"info": "No extractable text in PDF"}, ensure_ascii=False)
    else:
        stat = fp.stat()
        return json.dumps({
            "file_path": str(fp),
            "file_type": ext,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
            "info": "Binary file — content not directly readable. Use vault_semantic_search to find related content.",
        }, ensure_ascii=False, indent=2)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def vault_list_files(
    directory: str | None = None,
    file_type: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> str:
    """List files in the vault with optional directory and type filters.
    Results are paginated — use limit and offset to page through large directories.

    Args:
        directory: Optional subdirectory to list (relative to vault root)
        file_type: Optional filter: "text", "pdf", "image", "audio", "video"
        limit: Maximum number of files to return (default 200, max 1000)
        offset: Number of files to skip for pagination (default 0)

    Returns:
        JSON object with files list, total count, and pagination metadata
    """
    if file_type is not None and file_type not in _VALID_FILE_TYPES:
        return json.dumps(
            {"error": f"Invalid file_type '{file_type}'. Must be one of: {sorted(_VALID_FILE_TYPES)}"},
            ensure_ascii=False,
        )
    vault = VAULT_PATH
    if not vault or not str(vault):
        return json.dumps({"error": "VAULT_PATH is not set"}, ensure_ascii=False)

    base = vault / directory if directory else vault
    if not base.is_dir():
        return json.dumps({"error": f"Directory not found: {base}"}, ensure_ascii=False)

    if file_type and file_type in SUPPORTED_EXTENSIONS:
        exts = SUPPORTED_EXTENSIONS[file_type]
    else:
        exts = _TEXT_EXTS | _PDF_EXTS | _AUDIO_EXTS | _IMAGE_EXTS | _VIDEO_EXTS

    files = []
    for fp in sorted(base.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in exts:
            continue
        if any(part.startswith(".") for part in fp.relative_to(vault).parts):
            continue
        files.append(str(fp.relative_to(vault)))

    limit = min(max(1, limit), 1000)
    total = len(files)
    page = files[offset: offset + limit]
    return json.dumps(
        {
            "files": page,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": (offset + limit) < total,
            "next_offset": offset + limit if (offset + limit) < total else None,
        },
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

# Build a flat set of extensions handled in Phase 1
_TEXT_EXTS = SUPPORTED_EXTENSIONS["text"]
_PDF_EXTS = SUPPORTED_EXTENSIONS["pdf"]
_AUDIO_EXTS = SUPPORTED_EXTENSIONS["audio"]
_IMAGE_EXTS = SUPPORTED_EXTENSIONS["image"]
_VIDEO_EXTS = SUPPORTED_EXTENSIONS["video"]


def _ingest_single_file(file_path: Path) -> tuple[int, int]:
    """Index one file. Returns (chunk_count, error_count)."""
    ext = file_path.suffix.lower()
    fp_str = str(file_path)

    # Choose chunker
    if ext in _PDF_EXTS:
        chunks = chunk_pdf(file_path)
    elif ext in _AUDIO_EXTS:
        audio_chunks = chunk_audio(file_path)
        if not audio_chunks:
            return 0, 0
        # Audio uses native embedding (bytes), not text embedding
        index.delete_chunks_for_file(fp_str)
        errors = 0
        for ac in audio_chunks:
            chunk_id = f"{fp_str}::chunk_{ac['metadata']['chunk_index']}"
            try:
                embedding = embed_audio(ac["bytes"], ac["mime_type"])
                index.upsert_chunk(
                    file_path=fp_str,
                    chunk_id=chunk_id,
                    embedding=embedding,
                    chunk_text="",
                    file_type=ac["metadata"]["file_type"],
                    extra_metadata=ac["metadata"],
                )
            except Exception:
                logger.exception("Failed to embed/upsert audio chunk %s", chunk_id)
                errors += 1
        return len(audio_chunks) - errors, errors
    elif ext in _VIDEO_EXTS:
        video_chunks = chunk_video(file_path)
        if not video_chunks:
            return 0, 0
        # Video uses native embedding (bytes via embed_image — same Gemini API)
        index.delete_chunks_for_file(fp_str)
        errors = 0
        for vc in video_chunks:
            chunk_id = f"{fp_str}::chunk_{vc['metadata']['chunk_index']}"
            try:
                embedding = embed_image(vc["bytes"], vc["mime_type"])
                index.upsert_chunk(
                    file_path=fp_str,
                    chunk_id=chunk_id,
                    embedding=embedding,
                    chunk_text="",
                    file_type=vc["metadata"]["file_type"],
                    extra_metadata=vc["metadata"],
                )
            except Exception:
                logger.exception("Failed to embed/upsert video chunk %s", chunk_id)
                errors += 1
        return len(video_chunks) - errors, errors
    elif ext in _TEXT_EXTS:
        if ext == ".canvas":
            chunks = chunk_canvas(file_path)
        else:
            chunks = chunk_markdown(file_path)
    elif ext in _IMAGE_EXTS:
        image_chunks = chunk_image(file_path)
        if not image_chunks:
            return 0, 0
        # Image uses native embedding (bytes), not text embedding
        index.delete_chunks_for_file(fp_str)
        errors = 0
        for ic in image_chunks:
            chunk_id = f"{fp_str}::chunk_{ic['metadata']['chunk_index']}"
            try:
                embedding = embed_image(ic["bytes"], ic["mime_type"])
                index.upsert_chunk(
                    file_path=fp_str,
                    chunk_id=chunk_id,
                    embedding=embedding,
                    chunk_text="",
                    file_type=ic["metadata"]["file_type"],
                    extra_metadata=ic["metadata"],
                )
            except Exception:
                logger.exception("Failed to embed/upsert image chunk %s", chunk_id)
                errors += 1
        return len(image_chunks) - errors, errors
    else:
        return 0, 0

    if not chunks:
        return 0, 0

    # Remove stale chunks before re-indexing
    index.delete_chunks_for_file(fp_str)

    # Batch-embed all chunks in one API call
    try:
        embeddings = embed_texts([c["text"] for c in chunks])
    except Exception:
        logger.exception("Failed to batch-embed %d chunks for %s", len(chunks), fp_str)
        return 0, len(chunks)

    errors = 0
    for chunk, embedding in zip(chunks, embeddings):
        chunk_id = f"{fp_str}::chunk_{chunk['metadata']['chunk_index']}"
        try:
            index.upsert_chunk(
                file_path=fp_str,
                chunk_id=chunk_id,
                embedding=embedding,
                chunk_text=chunk["text"],
                file_type=chunk["metadata"]["file_type"],
                extra_metadata=chunk["metadata"],
            )
        except Exception:
            logger.exception("Failed to upsert chunk %s", chunk_id)
            errors += 1

    return len(chunks) - errors, errors


def ingest_file(file_path: Path) -> None:
    """Index a single file and print a summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    file_path = file_path.resolve()
    if not file_path.is_file():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Indexing {file_path} ...")
    indexed, errs = _ingest_single_file(file_path)
    print(f"Done — {indexed} chunks indexed, {errs} errors.")


def ingest_vault(vault_path: Path | None = None, *, mode: str = "full") -> None:
    """Walk the vault and index all supported files.

    Args:
        vault_path: Override vault path.
        mode: "md-only" (only .md), "text-only" (.md + .pdf), "full" (all types).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    vault = (vault_path or VAULT_PATH)
    if not vault or not str(vault):
        print("Error: VAULT_PATH is not set. Export VAULT_PATH or pass --vault")
        sys.exit(1)
    vault = vault.resolve()

    if mode == "md-only":
        all_exts = frozenset([".md"])
        print("Mode: md-only (markdown files only)")
    elif mode == "text-only":
        all_exts = frozenset([".md", ".canvas", ".pdf"])
        print("Mode: text-only (markdown, canvas, and PDF files)")
    else:
        all_exts = _TEXT_EXTS | _PDF_EXTS | _AUDIO_EXTS | _IMAGE_EXTS | _VIDEO_EXTS
        print("Mode: full (all supported file types)")
    total_files = 0
    total_chunks = 0
    total_errors = 0

    # Collect eligible files first so we can show progress
    eligible: list[Path] = []
    for fp in sorted(vault.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in all_exts:
            continue
        if any(part.startswith(".") for part in fp.relative_to(vault).parts):
            continue
        eligible.append(fp)

    total_eligible = len(eligible)
    workers = int(os.environ.get("INGEST_WORKERS", "16"))

    # Split into needs-reindex vs skip upfront (needs_reindex is cheap/local)
    to_index = [fp for fp in eligible if tracker.needs_reindex(fp)]
    skipped = total_eligible - len(to_index)
    print(
        f"Found {total_eligible} files: {len(to_index)} to index, "
        f"{skipped} unchanged. Workers: {workers}",
        flush=True,
    )

    start_time = time.monotonic()
    done_count = 0
    print_lock = threading.Lock()

    def _worker(fp: Path) -> tuple[int, int]:
        indexed, errs = _ingest_single_file(fp)
        if errs == 0:
            # Mark even 0-chunk files so they're skipped on future runs
            # (e.g. image-only PDFs, empty files) — only skip if no errors
            tracker.mark_indexed(fp)
        return indexed, errs

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, fp): fp for fp in to_index}
        for fut in as_completed(futures):
            fp = futures[fut]
            try:
                indexed, errs = fut.result()
            except Exception:
                logger.exception("Worker failed for %s", fp)
                indexed, errs = 0, 1
            with print_lock:
                done_count += 1
                total_chunks += indexed
                total_errors += errs
                elapsed = time.monotonic() - start_time
                rate = done_count / elapsed if elapsed > 0 else 0
                remaining = len(to_index) - done_count
                eta = remaining / rate if rate > 0 else float("inf")
                eta_str = f"{eta/60:.1f}min" if eta != float("inf") else "?"
                print(
                    f"  [{done_count}/{len(to_index)}] {fp.relative_to(vault)} "
                    f"({indexed} chunks) | {rate:.1f} files/s | ETA {eta_str}",
                    flush=True,
                )

    elapsed_total = time.monotonic() - start_time
    print(
        f"\nIngestion complete in {elapsed_total/60:.1f}min: "
        f"{total_eligible} files scanned, "
        f"{skipped} skipped (unchanged), "
        f"{done_count} indexed, "
        f"{total_chunks} chunks, {total_errors} errors.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def main():
    """Run the MCP server (stdio transport)."""
    mcp.run()


def ingest_cli():
    """CLI entry point for vault ingestion."""
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        description="Index Obsidian vault for semantic search",
    )
    parser.add_argument(
        "--vault", type=str, default=None, help="Vault path (overrides VAULT_PATH env)"
    )
    parser.add_argument(
        "--file", type=str, default=None, help="Index a single file instead of full vault"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--md-only", action="store_true", help="Only index markdown (.md) files"
    )
    mode_group.add_argument(
        "--text-only", action="store_true", help="Only index text-based files (.md, .canvas, .pdf)"
    )
    mode_group.add_argument(
        "--full", action="store_true", default=True, help="Index all supported file types (default)"
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Completely re-index from scratch. Backs up existing DB first; "
             "old DB is only deleted after successful rebuild."
    )
    args = parser.parse_args()

    vault = Path(args.vault) if args.vault else VAULT_PATH
    if not vault or not str(vault):
        print("Error: VAULT_PATH is not set. Export VAULT_PATH or pass --vault")
        sys.exit(1)
    if not vault.exists():
        print(f"Error: Vault path does not exist: {vault}")
        sys.exit(1)

    if args.file:
        ingest_file(Path(args.file))
        return

    # Determine mode
    mode = "full"
    if args.md_only:
        mode = "md-only"
    elif args.text_only:
        mode = "text-only"

    # Force rebuild: backup existing DB, clear index + tracker, then ingest
    if args.force_rebuild:
        from config import CHROMA_PERSIST_DIR
        backup_dir = None
        tracker_db = CHROMA_PERSIST_DIR.parent / "file_tracker.db"

        # Backup ChromaDB
        if CHROMA_PERSIST_DIR.exists():
            backup_dir = CHROMA_PERSIST_DIR.parent / "chroma.backup"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(CHROMA_PERSIST_DIR, backup_dir)
            print(f"Backed up ChromaDB to {backup_dir}")

        # Backup tracker DB
        tracker_backup = None
        if tracker_db.exists():
            tracker_backup = tracker_db.with_suffix(".db.backup")
            shutil.copy2(tracker_db, tracker_backup)
            print(f"Backed up tracker DB to {tracker_backup}")

        # Clear existing data
        try:
            if CHROMA_PERSIST_DIR.exists():
                shutil.rmtree(CHROMA_PERSIST_DIR)
            if tracker_db.exists():
                tracker_db.unlink()
            # Reset lazy-init so new collection is created
            import index as _idx
            _idx._client = None
            _idx._collection = None
            import tracker as _trk
            _trk._conn = None

            print("Cleared existing index. Starting fresh rebuild...")
            ingest_vault(vault, mode=mode)

            # Success — remove backups
            if backup_dir and backup_dir.exists():
                shutil.rmtree(backup_dir)
            if tracker_backup and tracker_backup.exists():
                tracker_backup.unlink()
            print("Rebuild successful. Backups removed.")
        except Exception as exc:
            # Restore from backup on failure
            print(f"\nRebuild FAILED: {exc}")
            if backup_dir and backup_dir.exists():
                if CHROMA_PERSIST_DIR.exists():
                    shutil.rmtree(CHROMA_PERSIST_DIR)
                shutil.copytree(backup_dir, CHROMA_PERSIST_DIR)
                shutil.rmtree(backup_dir)
                print("Restored ChromaDB from backup.")
            if tracker_backup and tracker_backup.exists():
                shutil.copy2(tracker_backup, tracker_db)
                tracker_backup.unlink()
                print("Restored tracker DB from backup.")
            sys.exit(1)
    else:
        ingest_vault(vault, mode=mode)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        sys.argv.pop(1)  # remove "ingest" from argv before argparse
        ingest_cli()
    else:
        main()
