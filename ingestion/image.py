"""Image ingestion for native Gemini multimodal embedding."""
import logging
from pathlib import Path
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Map file extensions to MIME types
MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
}


def chunk_image(file_path: str | Path) -> list[dict]:
    """Process an image file for embedding.

    Images are single-chunk: one embedding per image file.
    Normalizes to JPEG if needed (for size reduction), preserves PNG for transparency.

    Returns list with one dict:
        - "bytes": raw image bytes for Gemini embedding
        - "mime_type": MIME type string
        - "metadata": dict with file_path, chunk_index=0, file_type="image"
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    mime_type = MIME_MAP.get(ext, "image/jpeg")

    try:
        with open(file_path, "rb") as f:
            image_bytes = f.read()

        # Normalize: if file is too large (>10MB), resize
        if len(image_bytes) > 10 * 1024 * 1024:
            img = Image.open(io.BytesIO(image_bytes))
            img.thumbnail((2048, 2048))
            buf = io.BytesIO()
            fmt = "PNG" if ext == ".png" else "JPEG"
            img.save(buf, format=fmt)
            image_bytes = buf.getvalue()
            mime_type = "image/png" if ext == ".png" else "image/jpeg"
            logger.info("Resized large image %s to fit 10MB limit", file_path)

        return [{
            "bytes": image_bytes,
            "mime_type": mime_type,
            "metadata": {
                "file_path": str(file_path),
                "chunk_index": 0,
                "file_type": "image",
            }
        }]
    except Exception as e:
        logger.warning("Failed to process image %s: %s", file_path, e)
        return []
