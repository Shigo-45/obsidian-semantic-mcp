"""Gemini Embedding API wrapper with rate limiting."""
import time
import threading

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, EMBEDDING_DIM

_client = None
_client_lock = threading.Lock()


def _get_client():
    """Lazy-initialize the Gemini client on first use (thread-safe)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not GEMINI_API_KEY:
                    raise EnvironmentError(
                        "GEMINI_API_KEY is not set. "
                        "Export the environment variable before running: "
                        "export GEMINI_API_KEY='your-api-key'"
                    )
                _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# Token-bucket rate limiter — allows bursting up to MAX_TOKENS then refills at
# REFILL_RATE tokens/second.  Each embed_content call consumes 1 token.
# Default: 2000 RPM ≈ 33 RPS (conservative headroom below the 3000 RPM hard limit).
import os as _os

_REFILL_RATE: float = float(_os.environ.get("EMBED_RPS", "33"))  # tokens per second
_MAX_TOKENS: float = _REFILL_RATE  # burst up to 1 second's worth
_tokens: float = _MAX_TOKENS
_last_refill: float = time.monotonic()
_bucket_lock = threading.Lock()


def _rate_limit() -> None:
    """Consume one token from the bucket, blocking until one is available."""
    global _tokens, _last_refill
    while True:
        with _bucket_lock:
            now = time.monotonic()
            elapsed = now - _last_refill
            _tokens = min(_MAX_TOKENS, _tokens + elapsed * _REFILL_RATE)
            _last_refill = now
            if _tokens >= 1.0:
                _tokens -= 1.0
                return
            wait = (1.0 - _tokens) / _REFILL_RATE
        time.sleep(wait)


_BATCH_SIZE = 100  # Gemini batchEmbedContents limit


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Embed a single text string using the Gemini Embedding API."""
    return embed_texts([text], task_type=task_type)[0]


def embed_texts(
    texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:
    """Batch-embed a list of texts, splitting into _BATCH_SIZE chunks as needed.

    Returns one embedding vector per input text, in the same order.
    """
    results: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        _rate_limit()
        try:
            response = _get_client().models.embed_content(
                model=GEMINI_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            results.extend(list(emb.values) for emb in response.embeddings)
        except Exception as exc:
            raise RuntimeError(
                f"embed_texts failed for batch of {len(batch)} texts: {exc}"
            ) from exc
    return results


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
        result = _get_client().models.embed_content(
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
        result = _get_client().models.embed_content(
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
