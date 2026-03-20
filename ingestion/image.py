"""Image ingestion for native Gemini multimodal embedding."""
import io
import logging
from pathlib import Path

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()  # enables Image.open() for .heic/.heif

logger = logging.getLogger(__name__)

# Map file extensions to MIME types
MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/jpeg",  # converted to JPEG before sending
    ".heic": "image/jpeg",  # converted to JPEG before sending
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

        # HEIC/WebP: convert to JPEG (Gemini rejects these formats)
        if ext in (".heic", ".webp"):
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            image_bytes = buf.getvalue()
            mime_type = "image/jpeg"
            logger.info("Converted %s to JPEG: %s", ext, file_path)

        # Normalize: resize if file >10MB or either dimension >4096px
        else:
            img = Image.open(io.BytesIO(image_bytes))
            too_large = len(image_bytes) > 10 * 1024 * 1024
            too_wide = img.width > 4096 or img.height > 4096
            if too_large or too_wide:
                img.thumbnail((4096, 4096))
                buf = io.BytesIO()
                fmt = "PNG" if ext == ".png" else "JPEG"
                img.save(buf, format=fmt)
                image_bytes = buf.getvalue()
                mime_type = "image/png" if ext == ".png" else "image/jpeg"
                logger.info("Resized image %s (%dx%d)", file_path, img.width, img.height)

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
