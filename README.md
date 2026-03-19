# obsidian-semantic-mcp

A semantic search MCP server for Obsidian vaults. Indexes your vault content using Gemini embeddings and ChromaDB, then exposes it as MCP tools for Claude Code (or any MCP client).

## Features

- **Multi-file type support**: Markdown, Canvas, PDF, images, audio, video
- **Semantic search**: Natural language queries powered by Gemini embeddings
- **Incremental indexing**: SQLite mtime tracking — only re-indexes changed files
- **File watcher**: Optional watchdog-based auto-reindex on vault changes
- **5 MCP tools**: search, status, reindex, get file, list files

## Supported File Types

| Type | Extensions |
|------|-----------|
| Text | `.md`, `.canvas` |
| PDF | `.pdf` |
| Image | `.png`, `.jpg`, `.jpeg`, `.webp`, `.heic` |
| Audio | `.mp3`, `.m4a`, `.wav` |
| Video | `.mp4`, `.mov` |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone and install
git clone <repo-url> obsidian-semantic-mcp
cd obsidian-semantic-mcp
uv sync
```

Set environment variables:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export VAULT_PATH="/path/to/your/obsidian/vault"
```

## Usage

### Index your vault

```bash
# Full vault ingestion
uv run python server.py ingest --vault /path/to/vault

# Single file
uv run python server.py ingest --file /path/to/vault/note.md
```

### Run as MCP server

```bash
uv run python server.py
```

### Claude Code configuration

Add to your `.claude/mcp.json`:

```json
{
  "mcpServers": {
    "obsidian-semantic": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/obsidian-semantic-mcp", "python", "server.py"],
      "env": {
        "GEMINI_API_KEY": "your-key",
        "VAULT_PATH": "/path/to/vault"
      }
    }
  }
}
```

## MCP Tools

### `vault_semantic_search`

Search the vault using natural language semantic similarity.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural language search query |
| `n_results` | `int` | `10` | Number of results (max 50) |

### `vault_index_status`

Get the current index status: total chunks, file type counts, tracker stats.

### `vault_reindex`

Re-index the entire vault or a specific file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | `None` | Specific file to re-index. If omitted, re-indexes the full vault. |

### `vault_get_file`

Retrieve the full content of a vault file. Returns raw text for `.md`/`.canvas`, extracted text for PDFs, and metadata for binary files.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | required | Path to the file |

### `vault_list_files`

List files in the vault with optional directory and type filters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | `str` | `None` | Subdirectory relative to vault root |
| `file_type` | `str` | `None` | Filter: `"text"`, `"pdf"`, `"image"`, `"audio"`, `"video"` |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key |
| `VAULT_PATH` | Yes | — | Absolute path to Obsidian vault |
| `CHROMA_PERSIST_DIR` | No | `~/.obsidian-mcp/chroma` | ChromaDB storage location |
| `WATCH_VAULT` | No | `true` | Enable file watcher for auto-reindex |
