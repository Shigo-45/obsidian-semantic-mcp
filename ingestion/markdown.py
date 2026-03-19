"""Frontmatter-aware markdown chunker for Obsidian notes.

Parses YAML frontmatter and splits body text by H1/H2/H3 headings into
chunks suitable for embedding. Handles bilingual (Chinese + English) content
without any character-level transformations — pure line-based splitting
preserves UTF-8 / CJK text intact.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import frontmatter

logger = logging.getLogger(__name__)

# Heading pattern: lines that start with one to three '#' characters.
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)", re.MULTILINE)

# Maximum characters per chunk before paragraph-level splitting kicks in.
_MAX_CHUNK_CHARS = 2000


def _format_frontmatter(metadata: dict) -> str:
    """Render a frontmatter dict back to a YAML-fenced string."""
    lines = ["---"]
    for key, value in metadata.items():
        if isinstance(value, list):
            lines.append(f"{key}: [{', '.join(str(v) for v in value)}]")
        elif isinstance(value, dict):
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


def _split_by_paragraphs(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split *text* at double-newlines, re-joining short consecutive paragraphs
    until the combined length would exceed *max_chars*."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        join_overhead = 2 if current_parts else 0  # "\n\n" is 2 chars
        if current_parts and (current_len + join_overhead + para_len) > max_chars:
            chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_len = para_len
        else:
            current_parts.append(para)
            current_len += join_overhead + para_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def chunk_markdown(file_path: str | Path) -> list[dict]:
    """Parse an Obsidian markdown file and return a list of chunk dicts.

    Each chunk dict has the shape::

        {
            "text": str,          # chunk text, possibly prefixed with heading
            "metadata": {
                "file_path": str,
                "chunk_index": int,
                "file_type": "text",
                "heading": str | None,
                "frontmatter_tags": list | None,
            }
        }

    Chunking order:
    1. Frontmatter block (if non-empty).
    2. Body sections split by H1/H2/H3 headings; sections longer than
       ~2000 characters are further split at paragraph boundaries.
    """
    file_path = Path(file_path)
    try:
        post = frontmatter.load(str(file_path))
    except Exception as e:
        logger.warning("Failed to parse markdown file %s: %s", file_path, e)
        return []

    fm_dict: dict = dict(post.metadata)
    body: str = post.content  # body text without the YAML front matter
    tags = post.get("tags", None)

    chunks: list[dict] = []
    chunk_index = 0

    def _make_chunk(text: str, heading: str | None) -> dict:
        nonlocal chunk_index
        chunk = {
            "text": text,
            "metadata": {
                "file_path": str(file_path),
                "chunk_index": chunk_index,
                "file_type": "text",
                "heading": heading,
                "frontmatter_tags": tags,
            },
        }
        chunk_index += 1
        return chunk

    # ------------------------------------------------------------------ #
    # 1. Frontmatter chunk
    # ------------------------------------------------------------------ #
    if fm_dict:
        fm_text = _format_frontmatter(fm_dict)
        chunks.append(_make_chunk(fm_text, heading=None))

    # ------------------------------------------------------------------ #
    # 2. Body: split by headings
    # ------------------------------------------------------------------ #
    # Find all heading positions in the body text.
    heading_matches = list(_HEADING_RE.finditer(body))

    if not heading_matches:
        # No headings — treat entire body as one section.
        body_stripped = body.strip()
        if body_stripped:
            if len(body_stripped) > _MAX_CHUNK_CHARS:
                for part in _split_by_paragraphs(body_stripped):
                    chunks.append(_make_chunk(part, heading=None))
            else:
                chunks.append(_make_chunk(body_stripped, heading=None))
    else:
        # Text before the first heading (preamble).
        preamble = body[: heading_matches[0].start()].strip()
        if preamble:
            if len(preamble) > _MAX_CHUNK_CHARS:
                for part in _split_by_paragraphs(preamble):
                    chunks.append(_make_chunk(part, heading=None))
            else:
                chunks.append(_make_chunk(preamble, heading=None))

        for i, match in enumerate(heading_matches):
            heading_text = match.group(2).strip()
            section_start = match.start()
            section_end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(body)
            section_body = body[section_start:section_end].strip()

            if not section_body:
                continue

            if len(section_body) <= _MAX_CHUNK_CHARS:
                chunks.append(_make_chunk(section_body, heading=heading_text))
            else:
                # Keep the heading line in the first sub-chunk; subsequent
                # sub-chunks from the same section carry the heading in metadata only.
                heading_line = body[match.start() : match.end()]
                content_after_heading = body[match.end() : section_end].strip()
                parts = _split_by_paragraphs(content_after_heading)

                if parts:
                    first_part = heading_line + "\n" + parts[0]
                    chunks.append(_make_chunk(first_part, heading=heading_text))
                    for part in parts[1:]:
                        chunks.append(_make_chunk(part, heading=heading_text))
                else:
                    # Heading with no real body content.
                    chunks.append(_make_chunk(heading_line, heading=heading_text))

    return chunks
