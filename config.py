"""Configuration for obsidian-semantic-mcp."""
import os
from pathlib import Path

# Load .env file if present (same directory as this file)
_env_path = Path(__file__).parent / ".env"
if _env_path.is_file():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # Strip surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if value and key not in os.environ:  # don't override existing env vars
            os.environ[key] = value

# Vault settings — expand ~ in paths
VAULT_PATH = Path(os.environ.get("VAULT_PATH", "")).expanduser()

# ChromaDB persistence
CHROMA_PERSIST_DIR = Path(os.environ.get("CHROMA_PERSIST_DIR", str(Path.home() / ".obsidian-mcp" / "chroma"))).expanduser()

# Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 1536  # MRL 1536-dim (half of 3072 max) — Must match model's MRL output; update both GEMINI_MODEL and EMBEDDING_DIM together if model name changes

# Rate limiting — controls EMBEDDING throughput, NOT API calls.
# EMBED_RPM means "embeddings per minute". The actual API call interval is
# computed as:  60 / (EMBED_RPM / EMBED_BATCH_SIZE)
#
# Paid tier 1 (3K embeddings/min, 1M tokens/min, unlimited/day):
#   EMBED_RPM=3000, BATCH_SIZE=100 → 2s between API calls
# Free tier (~100-150 embeddings/min):
#   EMBED_RPM=100, BATCH_SIZE=50 → 30s between API calls
# For backwards compatibility, EMBED_RPS is read if EMBED_RPM is not set.
EMBED_RPM = float(os.environ.get(
    "EMBED_RPM",
    str(float(os.environ.get("EMBED_RPS", 2.0)) * 60)
))

# Batch size for text embedding API calls (Gemini limit: 100)
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "100"))

# Minimum batch size when adaptive backoff reduces it
EMBED_MIN_BATCH_SIZE = int(os.environ.get("EMBED_MIN_BATCH_SIZE", "10"))

# Batch size reduction factor on consecutive 429s (0.5 = halve each time)
EMBED_BACKOFF_BATCH_FACTOR = float(os.environ.get("EMBED_BACKOFF_BATCH_FACTOR", "0.5"))

# How many successful batches before restoring full batch size (0 = disabled)
EMBED_BACKOFF_RECOVERY_STEPS = int(os.environ.get("EMBED_BACKOFF_RECOVERY_STEPS", "5"))

# Retry configuration — exponential backoff with jitter
EMBED_MAX_RETRIES = int(os.environ.get("EMBED_MAX_RETRIES", "6"))
EMBED_RETRY_BASE_DELAY = float(os.environ.get("EMBED_RETRY_BASE_DELAY", "5"))
EMBED_RETRY_MAX_DELAY = float(os.environ.get("EMBED_RETRY_MAX_DELAY", "120"))

# Gemini also has opaque request/minute buckets such as
# global_embed_content_requests_per_minute_per_base_model / online_prediction_requests_per_base_model.
# Keep at least this many seconds between local Gemini embedding requests across MCP processes.
EMBED_MIN_REQUEST_INTERVAL = float(os.environ.get("EMBED_MIN_REQUEST_INTERVAL", "2"))

# Optional coarse token/byte budget. Text uses a ~4 chars/token estimate;
# multimodal native embeddings use uploaded byte size as a conservative proxy.
# Set below provider TPM because multimodal accounting is opaque and bursty.
EMBED_TPM = float(os.environ.get("EMBED_TPM", "250000"))

# Supported file types
SUPPORTED_EXTENSIONS = {
    "text": frozenset([".md", ".canvas"]),
    "pdf": frozenset([".pdf"]),
    "image": frozenset([".png", ".jpg", ".jpeg", ".webp", ".heic"]),
    "audio": frozenset([".mp3", ".m4a", ".wav"]),
    "video": frozenset([".mp4", ".mov"]),
}

# Audio segmentation (Gemini native audio limit)
AUDIO_MAX_SECONDS = 80

# Watch vault for changes
WATCH_VAULT = os.environ.get("WATCH_VAULT", "true").lower() == "true"

# Diagnostic logging for rate-limit analysis (429 debugging)
# Off by default — enable with EMBED_DIAG_LOG=true. Logs are structured JSON lines
# containing per-API-call metrics: timing, batch size, retry count, HTTP status,
# 429 response headers.  NEVER logs API keys, document bodies, or file paths.
EMBED_DIAG_LOG = os.environ.get("EMBED_DIAG_LOG", "false").lower() == "true"
EMBED_DIAG_LOG_FILE = os.environ.get("EMBED_DIAG_LOG_FILE", "")  # empty = stderr
