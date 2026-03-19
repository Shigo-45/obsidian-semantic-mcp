"""FastMCP server and CLI ingestion for obsidian-semantic-mcp."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import index
from config import SUPPORTED_EXTENSIONS, VAULT_PATH
from embedder import embed_text
from ingestion.markdown import chunk_markdown
from ingestion.pdf import chunk_pdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("obsidian-semantic")


@mcp.tool()
def vault_semantic_search(query: str, n_results: int = 10) -> str:
    """Search the vault using semantic similarity.

    Args:
        query: Natural language search query
        n_results: Number of results to return (default 10)

    Returns:
        JSON string with search results including file paths, scores, and snippets
    """
    n_results = min(n_results, 50)  # cap results
    try:
        embedding = embed_text(query, task_type="RETRIEVAL_QUERY")
        results = index.query(embedding, n_results=n_results)
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
    return json.dumps(status, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

# Build a flat set of extensions handled in Phase 1
_TEXT_EXTS = SUPPORTED_EXTENSIONS["text"]
_PDF_EXTS = SUPPORTED_EXTENSIONS["pdf"]


def _ingest_single_file(file_path: Path) -> tuple[int, int]:
    """Index one file. Returns (chunk_count, error_count)."""
    ext = file_path.suffix.lower()
    fp_str = str(file_path)

    # Choose chunker
    if ext in _PDF_EXTS:
        chunks = chunk_pdf(file_path)
    elif ext in _TEXT_EXTS:
        if ext == ".canvas":
            logger.info("Skipping .canvas (Phase 2): %s", fp_str)
            return 0, 0
        chunks = chunk_markdown(file_path)
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
    file_path = file_path.resolve()
    if not file_path.is_file():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Indexing {file_path} ...")
    indexed, errs = _ingest_single_file(file_path)
    print(f"Done — {indexed} chunks indexed, {errs} errors.")


def ingest_vault(vault_path: Path | None = None) -> None:
    """Walk the vault and index all supported files."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    vault = (vault_path or VAULT_PATH)
    if not vault or not str(vault):
        print("Error: VAULT_PATH is not set. Export VAULT_PATH or pass --vault")
        sys.exit(1)
    vault = vault.resolve()

    all_exts = _TEXT_EXTS | _PDF_EXTS
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

    for fp in eligible:
        total_files += 1
        indexed, errs = _ingest_single_file(fp)
        print(f"  [{total_files}/{total_eligible}] {fp.relative_to(vault)} ({indexed} chunks)")
        total_chunks += indexed
        total_errors += errs

    print(
        f"\nIngestion complete: {total_files} files processed, "
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

    parser = argparse.ArgumentParser(
        description="Index Obsidian vault for semantic search",
    )
    parser.add_argument(
        "--vault", type=str, default=None, help="Vault path (overrides VAULT_PATH env)"
    )
    parser.add_argument(
        "--file", type=str, default=None, help="Index a single file instead of full vault"
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
    else:
        ingest_vault(vault)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        sys.argv.pop(1)  # remove "ingest" from argv before argparse
        ingest_cli()
    else:
        main()
