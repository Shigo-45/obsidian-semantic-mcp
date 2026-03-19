"""Configuration for obsidian-semantic-mcp."""
import os
from pathlib import Path

# Vault settings
VAULT_PATH = Path(os.environ.get("VAULT_PATH", "/Users/shigo_45/Obsidian/45s-Zettelkasten"))

# ChromaDB persistence
CHROMA_PERSIST_DIR = Path(os.environ.get("CHROMA_PERSIST_DIR", Path.home() / ".obsidian-mcp" / "chroma"))

# Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIM = 1536  # MRL 1536-dim (half of 3072 max)

# Rate limiting (free tier: 100 RPM = ~1.67 RPS → 0.6s min between requests)
RATE_LIMIT_DELAY = 0.6  # seconds between API calls

# Supported file types
SUPPORTED_EXTENSIONS = {
    "text": [".md", ".canvas"],
    "pdf": [".pdf"],
    "image": [".png", ".jpg", ".jpeg", ".webp", ".heic"],
    "audio": [".mp3", ".m4a", ".wav"],
    "video": [".mp4", ".mov"],
}

# Audio segmentation (Gemini native audio limit)
AUDIO_MAX_SECONDS = 80

# Watch vault for changes
WATCH_VAULT = os.environ.get("WATCH_VAULT", "true").lower() == "true"
