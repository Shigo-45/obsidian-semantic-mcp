"""FastMCP server and CLI ingestion for obsidian-semantic-mcp."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import index
import tracker
from config import SUPPORTED_EXTENSIONS, VAULT_PATH
from embedder import embed_audio, embed_image, embed_text
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


@mcp.tool()
def vault_semantic_search(query: str, n_results: int = 10, file_type: str | None = None) -> str:
    """Search the Obsidian vault by meaning. Returns ranked results with file paths,
    relevance scores, and text snippets. Use this to find notes related to a topic
    even if they don't contain the exact search terms.

    Args:
        query: Natural language search query
        n_results: Number of results to return (default 10)
        file_type: Optional filter — "text" (md/canvas), "pdf", "image", "audio", "video"

    Returns:
        JSON string with search results including file paths, scores, and snippets
    """
    n_results = min(n_results, 50)  # cap results
    try:
        embedding = embed_text(query, task_type="RETRIEVAL_QUERY")
        results = index.query(embedding, n_results=n_results, file_type=file_type)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
def vault_index_status() -> str:
    """Get the current status of the vault index.

    Returns:
        JSON string with total chunks, counts by file type, and collection info
    """
    status = index.get_status()
    status["tracker"] = tracker.get_stats()
    return json.dumps(status, ensure_ascii=False, indent=2)


@mcp.tool()
def vault_reindex(file_path: str | None = None) -> str:
    """Re-index the vault or a specific file.

    Args:
        file_path: Optional path to a specific file. If None, re-indexes the entire vault.

    Returns:
        JSON string with indexing results
    """
    try:
        if file_path:
            fp = Path(file_path)
            if not fp.is_file():
                return json.dumps({"error": f"File not found: {file_path}"}, ensure_ascii=False)
            indexed, errors = _ingest_single_file(fp)
            if indexed > 0:
                tracker.mark_indexed(fp)
            return json.dumps({"file": file_path, "chunks_indexed": indexed, "errors": errors}, ensure_ascii=False)
        else:
            vault = VAULT_PATH
            if not vault or not str(vault):
                return json.dumps({"error": "VAULT_PATH is not set"}, ensure_ascii=False)
            ingest_vault(vault)
            status = index.get_status()
            return json.dumps({"status": "complete", **status}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
def vault_get_file(file_path: str) -> str:
    """Retrieve the full content of a file from the vault.

    For text files (.md, .canvas), returns the raw text content.
    For binary files (pdf, image, audio, video), returns metadata about the file.

    Args:
        file_path: Path to the file in the vault

    Returns:
        File content or metadata as a string
    """
    fp = Path(file_path)
    if not fp.is_file():
        return json.dumps({"error": f"File not found: {file_path}"}, ensure_ascii=False)

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
            "info": "Binary file — content not directly readable. Use vault_semantic_search to find related content."
        }, ensure_ascii=False, indent=2)


@mcp.tool()
def vault_list_files(directory: str | None = None, file_type: str | None = None) -> str:
    """List files in the vault with optional type filter.

    Args:
        directory: Optional subdirectory to list (relative to vault root)
        file_type: Optional filter: "text", "pdf", "image", "audio", "video"

    Returns:
        JSON list of file paths
    """
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

    return json.dumps(files, ensure_ascii=False, indent=2)


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

    errors = 0
    for chunk in chunks:
        chunk_id = f"{fp_str}::chunk_{chunk['metadata']['chunk_index']}"
        try:
            embedding = embed_text(chunk["text"])
            index.upsert_chunk(
                file_path=fp_str,
                chunk_id=chunk_id,
                embedding=embedding,
                chunk_text=chunk["text"],
                file_type=chunk["metadata"]["file_type"],
                extra_metadata=chunk["metadata"],
            )
        except Exception:
            logger.exception("Failed to embed/upsert chunk %s", chunk_id)
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
    print(f"Found {total_eligible} files to index.")

    skipped = 0
    for fp in eligible:
        total_files += 1
        if not tracker.needs_reindex(fp):
            skipped += 1
            continue
        indexed, errs = _ingest_single_file(fp)
        if indexed > 0:
            tracker.mark_indexed(fp)
        print(f"  [{total_files}/{total_eligible}] {fp.relative_to(vault)} ({indexed} chunks)")
        total_chunks += indexed
        total_errors += errs

    print(
        f"\nIngestion complete: {total_files} files processed, "
        f"{skipped} skipped (unchanged), "
        f"{total_chunks} chunks indexed, {total_errors} errors."
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
