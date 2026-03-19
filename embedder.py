"""Gemini Embedding API wrapper with rate limiting."""
import time
import threading

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, EMBEDDING_DIM, RATE_LIMIT_DELAY

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. "
        "Export the environment variable before running: "
        "export GEMINI_API_KEY='your-api-key'"
    )

client = genai.Client(api_key=GEMINI_API_KEY)

# Rate limiter state
_last_call_time: float = 0.0
_lock = threading.Lock()


def _rate_limit() -> None:
    """Enforce at least RATE_LIMIT_DELAY seconds between API calls."""
    global _last_call_time
    with _lock:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        _last_call_time = time.time()


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Embed a text string using the Gemini Embedding API.

    Args:
        text: The text to embed.
        task_type: Gemini task type (e.g. "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY",
                   "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING").

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        RuntimeError: If the API call fails or returns unexpected structure.
    """
    _rate_limit()
    try:
        result = client.models.embed_content(
            model=GEMINI_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        return list(result.embeddings[0].values)
    except Exception as exc:
        raise RuntimeError(
            f"embed_text failed for text of length {len(text)}: {exc}"
        ) from exc


def embed_image(image_bytes: bytes, mime_type: str) -> list[float]:
    """Embed image bytes natively using the Gemini Embedding API.

    Args:
        image_bytes: Raw image bytes.
        mime_type: MIME type of the image (e.g. "image/jpeg", "image/png").

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        RuntimeError: If the API call fails or returns unexpected structure.
    """
    _rate_limit()
    try:
        result = client.models.embed_content(
            model=GEMINI_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        return list(result.embeddings[0].values)
    except Exception as exc:
        raise RuntimeError(
            f"embed_image failed for {mime_type} image of {len(image_bytes)} bytes: {exc}"
        ) from exc


def embed_audio(audio_bytes: bytes, mime_type: str) -> list[float]:
    """Embed audio bytes natively using the Gemini Embedding API.

    Audio must be at most AUDIO_MAX_SECONDS (80 s) long — segmentation is the
    caller's responsibility.

    Args:
        audio_bytes: Raw audio bytes.
        mime_type: MIME type of the audio (e.g. "audio/mp3", "audio/wav").

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        RuntimeError: If the API call fails or returns unexpected structure.
    """
    _rate_limit()
    try:
        result = client.models.embed_content(
            model=GEMINI_MODEL,
            contents=[types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)],
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        return list(result.embeddings[0].values)
    except Exception as exc:
        raise RuntimeError(
            f"embed_audio failed for {mime_type} audio of {len(audio_bytes)} bytes: {exc}"
        ) from exc
