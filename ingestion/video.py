"""Video ingestion via Gemini File API for native embedding."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIME_MAP = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
}

# Max video size for inline embedding (20MB)
MAX_VIDEO_SIZE = 20 * 1024 * 1024


def chunk_video(file_path: str | Path) -> list[dict]:
    """Process a video file for embedding.

    Videos are single-chunk: one embedding per video file.
    Uses raw bytes for files <= 20MB.

    Returns list with one dict:
        - "bytes": raw video bytes
        - "mime_type": MIME type string
        - "metadata": dict with file_path, chunk_index=0, file_type="video"
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    mime_type = MIME_MAP.get(ext, "video/mp4")

    try:
        file_size = file_path.stat().st_size
        if file_size > MAX_VIDEO_SIZE:
            logger.warning(
                "Video %s is %d MB, exceeds 20MB limit — skipping",
                file_path,
                file_size // (1024 * 1024),
            )
            return []

        with open(file_path, "rb") as f:
            video_bytes = f.read()

        return [
            {
                "bytes": video_bytes,
                "mime_type": mime_type,
                "metadata": {
                    "file_path": str(file_path),
                    "chunk_index": 0,
                    "file_type": "video",
                },
            }
        ]
    except Exception as e:
        logger.warning("Failed to process video %s: %s", file_path, e)
        return []
