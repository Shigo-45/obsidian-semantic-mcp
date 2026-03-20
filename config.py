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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIM = 1536  # MRL 1536-dim (half of 3072 max) — Must match model's MRL output; update both GEMINI_MODEL and EMBEDDING_DIM together if model name changes

# Rate limiting — paid tier allows higher throughput (2000 RPM)
# Free tier: 0.6s (100 RPM), Paid tier: 0.05s (1200 RPM, conservative)
RATE_LIMIT_DELAY = float(os.environ.get("RATE_LIMIT_DELAY", "0.05"))

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
