"""Obsidian .canvas file ingestion."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def chunk_canvas(file_path: str | Path) -> list[dict]:
    """Parse an Obsidian .canvas file and extract text content.

    Canvas files are JSON with "nodes" (text cards, file links, URLs) and "edges" (connections).
    Extracts text from all nodes into chunks.

    Returns list of dicts with:
        - "text": node text content
        - "metadata": dict with file_path, chunk_index, file_type="text", node_type, node_id
    """
    file_path = Path(file_path)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to parse canvas %s: %s", file_path, e)
        return []

    chunks = []
    chunk_index = 0
    nodes = data.get("nodes", [])

    for node in nodes:
        node_type = node.get("type", "")
        text = ""

        if node_type == "text":
            text = node.get("text", "").strip()
        elif node_type == "file":
            # File reference node — include the file path as searchable text
            text = f"File link: {node.get('file', '')}"
        elif node_type == "link":
            text = f"URL: {node.get('url', '')}"
        elif node_type == "group":
            label = node.get("label", "").strip()
            if label:
                text = f"Group: {label}"

        if not text:
            continue

        chunks.append({
            "text": text,
            "metadata": {
                "file_path": str(file_path),
                "chunk_index": chunk_index,
                "file_type": "text",  # canvas text is embedded as text
                "node_type": node_type,
                "node_id": node.get("id", ""),
            }
        })
        chunk_index += 1

    # Also extract edge labels if present
    edges = data.get("edges", [])
    for edge in edges:
        label = edge.get("label", "").strip()
        if label:
            chunks.append({
                "text": f"Connection: {label}",
                "metadata": {
                    "file_path": str(file_path),
                    "chunk_index": chunk_index,
                    "file_type": "text",
                    "node_type": "edge",
                    "node_id": edge.get("id", ""),
                }
            })
            chunk_index += 1

    return chunks
