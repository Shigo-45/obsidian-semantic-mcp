"""File system watcher for auto-reindexing on vault changes."""
import logging
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

_all_exts: frozenset[str] = frozenset()
for _exts in SUPPORTED_EXTENSIONS.values():
    _all_exts = _all_exts | _exts


class VaultHandler(FileSystemEventHandler):
    """Handle file changes in the vault."""

    def __init__(self, on_change):
        self._on_change = on_change

    def _should_handle(self, path: str) -> bool:
        p = Path(path)
        if any(part.startswith(".") for part in p.parts):
            return False
        return p.suffix.lower() in _all_exts

    def on_created(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            logger.info("File created: %s", event.src_path)
            self._on_change(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            logger.info("File modified: %s", event.src_path)
            self._on_change(Path(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            logger.info("File deleted: %s", event.src_path)
            import index
            import tracker
            index.delete_chunks_for_file(event.src_path)
            tracker.remove_file(event.src_path)


def start_watcher(vault_path: Path, on_change) -> Observer:
    """Start watching the vault for file changes.

    Args:
        vault_path: Path to the vault directory
        on_change: Callback function(file_path: Path) called when a file changes

    Returns:
        The Observer instance (call .stop() to stop watching)
    """
    handler = VaultHandler(on_change)
    observer = Observer()
    observer.schedule(handler, str(vault_path), recursive=True)
    observer.start()
    logger.info("Watching vault at %s for changes", vault_path)
    return observer
