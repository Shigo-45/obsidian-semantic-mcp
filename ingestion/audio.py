"""Audio ingestion with segment splitting for Gemini native embedding."""

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_MAX_MS = 80 * 1000  # 80 seconds in milliseconds

MIME_MAP = {
    ".mp3": "audio/mp3",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
}


def chunk_audio(file_path: str | Path) -> list[dict]:
    """Process an audio file for embedding, splitting if >80s.

    Returns a list of chunk dicts, each with ``bytes``, ``mime_type``,
    and ``metadata`` keys.  If the file is <= 80 s the original bytes
    are returned as a single chunk; otherwise the audio is split into
    80-second segments exported as mp3.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(str(file_path))
    except Exception as e:
        logger.warning("Failed to load audio %s: %s", file_path, e)
        return []

    duration_ms = len(audio)
    chunks: list[dict] = []

    if duration_ms <= AUDIO_MAX_MS:
        # Short audio — use original bytes
        mime_type = MIME_MAP.get(ext, "audio/mp3")
        try:
            audio_bytes = file_path.read_bytes()
        except Exception as e:
            logger.warning("Failed to read audio file %s: %s", file_path, e)
            return []

        chunks.append({
            "bytes": audio_bytes,
            "mime_type": mime_type,
            "metadata": {
                "file_path": str(file_path),
                "chunk_index": 0,
                "file_type": "audio",
                "segment_start_s": 0,
                "segment_end_s": round(duration_ms / 1000, 1),
            },
        })
    else:
        # Long audio — split into 80s segments, export as mp3
        chunk_index = 0
        start_ms = 0
        while start_ms < duration_ms:
            end_ms = min(start_ms + AUDIO_MAX_MS, duration_ms)
            segment = audio[start_ms:end_ms]

            buf = io.BytesIO()
            try:
                segment.export(buf, format="mp3")
            except Exception as e:
                logger.warning(
                    "Failed to export audio segment %d of %s: %s",
                    chunk_index, file_path, e,
                )
                start_ms = end_ms
                chunk_index += 1
                continue

            chunks.append({
                "bytes": buf.getvalue(),
                "mime_type": "audio/mp3",
                "metadata": {
                    "file_path": str(file_path),
                    "chunk_index": chunk_index,
                    "file_type": "audio",
                    "segment_start_s": round(start_ms / 1000, 1),
                    "segment_end_s": round(end_ms / 1000, 1),
                },
            })
            start_ms = end_ms
            chunk_index += 1

    return chunks
