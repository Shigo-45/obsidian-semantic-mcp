"""PDF ingestion for obsidian-semantic-mcp.

Extracts text from PDF files page by page using pdfplumber.
Each page becomes one chunk for page-level attribution.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def chunk_pdf(file_path: str | Path) -> list[dict]:
    """Extract text from a PDF file, returning one chunk per page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of chunk dicts, each with:
            - text (str): Page text content.
            - metadata (dict): file_path, chunk_index, file_type, page_number,
              total_pages.

        Pages with no extractable text (e.g. image-only pages) are skipped.
        If the PDF cannot be opened, logs a warning and returns an empty list.
    """
    import pdfplumber  # local import so the module is importable without pdfplumber

    file_path = Path(file_path)
    chunks: list[dict] = []

    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            chunk_index = 0
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                chunks.append(
                    {
                        "text": text,
                        "metadata": {
                            "file_path": str(file_path),
                            "chunk_index": chunk_index,
                            "file_type": "pdf",
                            "page_number": i + 1,  # 1-indexed
                            "total_pages": total_pages,
                        },
                    }
                )
                chunk_index += 1
    except Exception as exc:
        logger.warning("Failed to open PDF %s: %s", file_path, exc)
        return []

    return chunks
