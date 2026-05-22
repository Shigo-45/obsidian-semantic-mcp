"""Gemini Embedding API wrapper with rate limiting and 429 backoff."""
import json
import os
import random
import sys
import tempfile
import threading
import time
from pathlib import Path
from datetime import datetime, timezone

from google import genai
from google.genai import types
from google.genai.errors import APIError

from config import (
    EMBED_BACKOFF_BATCH_FACTOR,
    EMBED_BACKOFF_RECOVERY_STEPS,
    EMBED_BATCH_SIZE,
    EMBED_DIAG_LOG,
    EMBED_DIAG_LOG_FILE,
    EMBED_MAX_RETRIES,
    EMBED_MIN_BATCH_SIZE,
    EMBED_RETRY_BASE_DELAY,
    EMBED_RETRY_MAX_DELAY,
    EMBED_RPM,
    EMBED_TPM,
    EMBEDDING_DIM,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

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


# Global request pacing.  Gemini/AI Studio quotas are evaluated per embedding
# (not per API call).  EMBED_RPM means "embeddings per minute" — the API-call
# interval is auto-computed as:  60 / (EMBED_RPM / batch_size)
#
# Adaptive batch sizing: on consecutive 429 errors the batch size is reduced
# (down to EMBED_MIN_BATCH_SIZE) and gradually restored after enough successes.
import os as _os

_MIN_INTERVAL_PER_EMBEDDING = 60.0 / max(EMBED_RPM, 0.001)
_last_request: float = 0.0
_rate_lock = threading.Lock()

# Cross-process pacing state. Obsidian MCP servers, Zotero MCP, and nightly
# workers can all share the same Gemini base-model quota. A per-process limiter
# still bursts when several MCP processes run at once, so serialize the timestamp
# through a small file lock keyed by model name.
_RATE_STATE_DIR = Path(os.environ.get("EMBED_RATE_STATE_DIR", tempfile.gettempdir()))
_RATE_STATE_PATH = _RATE_STATE_DIR / f"hermes-gemini-embed-{GEMINI_MODEL.replace('/', '_')}.state"
_MIN_INTERVAL_PER_TOKEN = 60.0 / max(EMBED_TPM, 0.001)


def _rate_limit_cross_process(n_embeddings: int) -> None:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX fallback
        return _rate_limit_process_only(n_embeddings)

    _RATE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    interval = _MIN_INTERVAL_PER_EMBEDDING * max(1, n_embeddings)
    with open(_RATE_STATE_PATH, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            raw = fh.read().strip()
            try:
                last = float(raw) if raw else 0.0
            except ValueError:
                last = 0.0
            now = time.monotonic()
            wait = interval - (now - last)
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            fh.seek(0)
            fh.truncate()
            fh.write(str(now))
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _call_with_cross_process_pacing(fn, n_embeddings: int, est_tokens: int = 0):
    """Run one Gemini API call while holding the shared quota lock.

    The server quota is enforced on overlapping in-flight requests too. Holding
    the lock until the response returns prevents several local MCP processes from
    starting many large multimodal embedding calls simultaneously.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX fallback
        _rate_limit_process_only(n_embeddings)
        return fn()

    _RATE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    interval = max(
        _MIN_INTERVAL_PER_EMBEDDING * max(1, n_embeddings),
        _MIN_INTERVAL_PER_TOKEN * max(0, est_tokens),
    )
    with open(_RATE_STATE_PATH, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            raw = fh.read().strip()
            try:
                last = float(raw) if raw else 0.0
            except ValueError:
                last = 0.0
            now = time.monotonic()
            wait = interval - (now - last)
            if wait > 0:
                time.sleep(wait)
            result = fn()
            fh.seek(0)
            fh.truncate()
            fh.write(str(time.monotonic()))
            fh.flush()
            return result
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _rate_limit_process_only(n_embeddings: int = 1) -> None:
    """Process-local fallback for platforms without fcntl."""
    global _last_request
    with _rate_lock:
        now = time.monotonic()
        wait = (_MIN_INTERVAL_PER_EMBEDDING * max(1, n_embeddings)) - (now - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()

# Adaptive state — guarded by _rate_lock
_current_batch_size: int = EMBED_BATCH_SIZE
_consecutive_429s: int = 0
_consecutive_successes: int = 0


def _rate_limit(n_embeddings: int = 1) -> None:
    """Serialize embed API calls across processes by *embedding count*.

    Gemini embedding quota is shared per base model/project, so all local MCP
    processes must share one pacing clock. EMBED_RPM means embeddings/minute,
    not API calls/minute: a 100-text batch at EMBED_RPM=3000 waits ~2s.
    """
    _rate_limit_cross_process(max(1, n_embeddings))


def _on_429() -> None:
    """Reduce batch size and reset success counter on a 429 response."""
    global _current_batch_size, _consecutive_429s, _consecutive_successes
    with _rate_lock:
        _consecutive_429s += 1
        _consecutive_successes = 0
        _current_batch_size = max(
            EMBED_MIN_BATCH_SIZE,
            int(_current_batch_size * EMBED_BACKOFF_BATCH_FACTOR),
        )


def _on_success() -> None:
    """Increment success counter; restore full batch size after enough successes."""
    global _consecutive_successes, _consecutive_429s, _current_batch_size
    with _rate_lock:
        _consecutive_successes += 1
        if (
            EMBED_BACKOFF_RECOVERY_STEPS > 0
            and _consecutive_successes >= EMBED_BACKOFF_RECOVERY_STEPS
        ):
            _current_batch_size = EMBED_BATCH_SIZE
            _consecutive_429s = 0
            _consecutive_successes = 0


def diag_log_config_snapshot() -> None:
    """Write a one-time config snapshot to the diagnostic log.

    Exposes every rate-limit and concurrency knob so you can correlate log
    entries with the exact configuration that produced them.  Safe to call
    multiple times (no-ops when EMBED_DIAG_LOG is off).
    """
    import os as _os2

    ingest_workers = int(_os2.environ.get("INGEST_WORKERS", "2"))
    _diag_write(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "config_snapshot",
            "config": {
                "GEMINI_MODEL": GEMINI_MODEL,
                "EMBEDDING_DIM": EMBEDDING_DIM,
                "EMBED_RPM": EMBED_RPM,
                "EMBED_TPM": EMBED_TPM,
                "EMBED_BATCH_SIZE": EMBED_BATCH_SIZE,
                "EMBED_MIN_BATCH_SIZE": EMBED_MIN_BATCH_SIZE,
                "EMBED_BACKOFF_BATCH_FACTOR": EMBED_BACKOFF_BATCH_FACTOR,
                "EMBED_BACKOFF_RECOVERY_STEPS": EMBED_BACKOFF_RECOVERY_STEPS,
                "EMBED_MAX_RETRIES": EMBED_MAX_RETRIES,
                "EMBED_RETRY_BASE_DELAY": f"{EMBED_RETRY_BASE_DELAY}s",
                "EMBED_RETRY_MAX_DELAY": f"{EMBED_RETRY_MAX_DELAY}s",
                "INGEST_WORKERS": ingest_workers,
            },
        }
    )


# ---------------------------------------------------------------------------
# Diagnostic logging for rate-limit analysis (429 debugging)
# ---------------------------------------------------------------------------
# Controlled by EMBED_DIAG_LOG / EMBED_DIAG_LOG_FILE in config.py.
# Writes one JSON object per line — easy to grep / jq / analyze.
# NEVER logs API keys, document body text, or file paths.

_diag_file = None
_diag_lock = threading.Lock()


def _diag_write(entry: dict) -> None:
    """Write a diagnostic entry as a JSON line (thread-safe)."""
    global _diag_file
    if not EMBED_DIAG_LOG:
        return
    with _diag_lock:
        if _diag_file is None:
            if EMBED_DIAG_LOG_FILE:
                _diag_file = open(EMBED_DIAG_LOG_FILE, "a", encoding="utf-8")
            else:
                _diag_file = sys.stderr
        line = json.dumps(entry, ensure_ascii=False, default=str)
        _diag_file.write(line + "\n")
        _diag_file.flush()


def _extract_429_headers(exc: Exception) -> dict[str, str]:
    """Pull rate-limit-relevant headers from a 429 APIError response.

    The google-genai SDK stores the raw httpx Response on APIError.response,
    so we can extract Retry-After, X-RateLimit-*, etc. without scraping the
    error message string.  Returns an empty dict when headers aren't available
    (non-APIError, missing response object, etc.).
    """
    if not isinstance(exc, APIError):
        return {}
    try:
        resp = exc.response  # type: ignore[attr-defined]
    except AttributeError:
        return {}
    if resp is None:
        return {}
    headers = {}
    for key, value in resp.headers.items():
        kl = key.lower()
        if any(
            tag in kl
            for tag in (
                "retry-after",
                "ratelimit",
                "rate-limit",
                "x-goog-",
                "x-ratelimit",
            )
        ):
            headers[key] = value
    return headers


def _truncate_error(exc: Exception, max_len: int = 300) -> str:
    """Safe, truncated error string — no API keys, no document bodies."""
    msg = str(exc)
    return msg[:max_len] if len(msg) > max_len else msg


def _is_retryable_429(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _parse_retry_after(headers: dict[str, str]) -> float | None:
    """Parse Retry-After header from 429 response headers dict.

    Supports both delta-seconds (integer) and HTTP-date formats.
    Returns seconds to wait, or None if not found / unparseable.
    """
    for key, value in headers.items():
        kl = key.lower()
        if "retry-after" not in kl:
            continue
        # Try delta-seconds first (e.g. "120")
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
        # Try HTTP-date format (e.g. "Wed, 21 Oct 2015 07:28:00 GMT")
        try:
            from email.utils import parsedate_to_datetime
            retry_dt = parsedate_to_datetime(value)
            return max(0.0, (retry_dt - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            pass
    return None


def _sleep_backoff(attempt: int, min_delay: float = 0.0) -> None:
    """Sleep with exponential backoff + jitter, respecting a minimum delay."""
    delay = min(EMBED_RETRY_BASE_DELAY * (2 ** attempt), EMBED_RETRY_MAX_DELAY)
    delay = max(delay, min_delay)
    jitter = random.uniform(0.8, 1.2)
    time.sleep(delay * jitter)


def _call_with_retry(
    fn,
    *,
    kind: str,
    batch_size: int,
    est_tokens: int = 0,
) -> list[list[float]] | list[float]:
    """Execute ``fn()`` with up to EMBED_MAX_RETRIES retries on 429.

    - Respects the Retry-After header from 429 responses as a minimum delay.
    - Triggers adaptive batch-size reduction on consecutive 429s.
    - Gradually restores batch size after enough successes.
    - When EMBED_DIAG_LOG is enabled, writes a JSON diagnostic entry for each
      call containing timing, batch size, retry count, HTTP status, and — on
      429 — relevant response headers.
    """
    t0 = time.monotonic()
    last_exc: Exception | None = None
    for attempt in range(EMBED_MAX_RETRIES + 1):
        try:
            result = _call_with_cross_process_pacing(fn, batch_size, est_tokens)
            # Success — log, notify adaptive state, return
            _on_success()
            _diag_write(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "kind": kind,
                    "model": GEMINI_MODEL,
                    "batch_size": batch_size,
                    "adaptive_batch_size": _current_batch_size,
                    "consecutive_429s": _consecutive_429s,
                    "est_tokens": est_tokens,
                    "duration_ms": round((time.monotonic() - t0) * 1000),
                    "retry_count": attempt,
                    "http_status": 200,
                    "success": True,
                }
            )
            return result
        except Exception as exc:
            last_exc = exc
            is_429 = _is_retryable_429(exc)
            http_status = 429 if is_429 else getattr(exc, "code", 0)
            if not is_429 or attempt >= EMBED_MAX_RETRIES:
                # Final failure
                if is_429:
                    _on_429()
                entry: dict = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "kind": kind,
                    "model": GEMINI_MODEL,
                    "batch_size": batch_size,
                    "adaptive_batch_size": _current_batch_size,
                    "consecutive_429s": _consecutive_429s,
                    "est_tokens": est_tokens,
                    "duration_ms": round((time.monotonic() - t0) * 1000),
                    "retry_count": attempt,
                    "http_status": int(http_status) if http_status else 0,
                    "success": False,
                    "error_msg": _truncate_error(exc),
                }
                if is_429:
                    headers_429 = _extract_429_headers(exc)
                    if headers_429:
                        entry["429_headers"] = headers_429
                _diag_write(entry)
                raise

            # Retryable 429 — reduce batch size for future calls,
            # then backoff respecting Retry-After if present.
            _on_429()
            headers_429 = _extract_429_headers(exc)
            retry_after = _parse_retry_after(headers_429) if headers_429 else None
            min_delay = retry_after if retry_after is not None else 0.0

            # Log each retry as a mid-flight diagnostic entry
            entry_retry: dict = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                "model": GEMINI_MODEL,
                "batch_size": batch_size,
                "adaptive_batch_size": _current_batch_size,
                "consecutive_429s": _consecutive_429s,
                "est_tokens": est_tokens,
                "duration_ms": round((time.monotonic() - t0) * 1000),
                "retry_count": attempt + 1,
                "http_status": 429,
                "success": False,
                "phase": "retry",
                "error_msg": _truncate_error(exc),
                "retry_after_s": retry_after,
            }
            if headers_429:
                entry_retry["429_headers"] = headers_429
            _diag_write(entry_retry)
            _sleep_backoff(attempt, min_delay=min_delay)
    assert last_exc is not None
    raise RuntimeError(
        f"{kind} failed for batch of {batch_size}: {last_exc}"
    ) from last_exc


# gemini-embedding-2 uses prompt prefixes instead of task_type parameter.
# See: https://ai.google.dev/gemini-api/docs/embeddings
_QUERY_PREFIX = "task: search result | query: "
_DOC_PREFIX = "title: none | text: "


def _format_for_embedding(text: str, task: str) -> str:
    """Format text with gemini-embedding-2 task-specific prompt prefix."""
    if task == "query":
        return f"{_QUERY_PREFIX}{text}"
    elif task == "document":
        return f"{_DOC_PREFIX}{text}"
    else:
        return text  # passthrough for raw/other uses


def embed_text(text: str, task: str = "document") -> list[float]:
    """Embed a single text string using the Gemini Embedding API."""
    return embed_texts([text], task=task)[0]


def embed_texts(
    texts: list[str], task: str = "document"
) -> list[list[float]]:
    """Batch-embed a list of texts, splitting by adaptive batch size.

    Returns one embedding vector per input text, in the same order.
    Batch size is dynamically reduced on 429s and restored on success.
    Rate limiting counts EMBEDDINGS (not API calls) against EMBED_RPM.
    """
    formatted = [_format_for_embedding(t, task) for t in texts]
    results: list[list[float]] = []
    batch_size = _current_batch_size  # snapshot for this call
    for i in range(0, len(formatted), batch_size):
        batch = formatted[i : i + batch_size]
        try:
            # Estimate token count: ~4 chars per token for English text,
            # plus prompt-prefix overhead.  Not exact — proportional indicator.
            est = sum(len(t) for t in batch) // 4
            response = _call_with_retry(
                lambda: _get_client().models.embed_content(
                    model=GEMINI_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=EMBEDDING_DIM,
                    ),
                ),
                kind="embed_texts",
                batch_size=len(batch),
                est_tokens=est,
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
    try:
        result = _call_with_retry(
            lambda: _get_client().models.embed_content(
                model=GEMINI_MODEL,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIM,
                ),
            ),
            kind="embed_image",
            batch_size=1,
            est_tokens=len(image_bytes),
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
    try:
        result = _call_with_retry(
            lambda: _get_client().models.embed_content(
                model=GEMINI_MODEL,
                contents=[types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)],
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIM,
                ),
            ),
            kind="embed_audio",
            batch_size=1,
            est_tokens=len(audio_bytes),
        )
        return list(result.embeddings[0].values)
    except Exception as exc:
        raise RuntimeError(
            f"embed_audio failed for {mime_type} audio of {len(audio_bytes)} bytes: {exc}"
        ) from exc
