"""Tests for index.get_status() and index.get_indexed_files() batched fetching.

Run with:  python -m pytest tests/test_index.py -v
Or plain:  python tests/test_index.py
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGetStatusBatching(unittest.TestCase):
    """index.get_status() must use paginated get() to avoid SQLite variable-limit errors."""

    def _make_meta(self, file_type):
        return {"file_path": f"/vault/{file_type}/note.md", "file_type": file_type}

    def test_small_index_returns_correct_counts(self):
        """With a small index, get_status should aggregate file types correctly."""
        col = Mock()
        col.name = "vault"
        col.count.return_value = 5
        col.get.return_value = {
            "metadatas": [
                self._make_meta("text"),
                self._make_meta("text"),
                self._make_meta("pdf"),
                self._make_meta("text"),
                self._make_meta("image"),
            ]
        }

        with patch("index._get_collection", return_value=col):
            from index import get_status

            status = get_status()

        self.assertEqual(status["total_chunks"], 5)
        self.assertEqual(status["by_file_type"], {"text": 3, "pdf": 1, "image": 1})
        self.assertEqual(status["collection_name"], "vault")

        # Must use limit and offset, not bare get()
        col.get.assert_called_once()
        _, kwargs = col.get.call_args
        self.assertIn("limit", kwargs)
        self.assertIn("offset", kwargs)

    def test_batches_large_index_across_multiple_calls(self):
        """When the index has more than _GET_BATCH_SIZE chunks, get_status must
        iterate with multiple get() calls and accumulate correctly."""
        import index as idx_mod

        B = idx_mod._GET_BATCH_SIZE
        total_chunks = B + B + 7  # 2 full batches + 1 partial

        # Create deterministic metadata lists
        metas_full = [
            self._make_meta("text") for _ in range(B)
        ]
        metas_partial = [
            self._make_meta("pdf") for _ in range(B + 7)
        ]

        col = Mock()
        col.name = "vault"
        col.count.return_value = total_chunks

        # Side effect: batch 0 → full text, batch 1 → B pdf chunks, batch 2 → 7 pdf chunks
        call_count = [0]

        def get_side_effect(**kwargs):
            offset = kwargs.get("offset", 0)
            limit = kwargs.get("limit", B)
            call_count[0] += 1

            start = offset
            end = min(offset + limit, total_chunks)

            if start < B:
                # First batch: all text
                chunk = metas_full[start:end]
            else:
                # Second + third batches: all pdf
                chunk = metas_partial[start - B : end - B]

            return {"metadatas": chunk}

        col.get.side_effect = get_side_effect

        with patch("index._get_collection", return_value=col):
            from index import get_status

            status = get_status()

        self.assertEqual(status["total_chunks"], total_chunks)
        self.assertEqual(status["by_file_type"], {"text": B, "pdf": B + 7})
        # Must have made multiple get() calls
        self.assertGreater(call_count[0], 1)

    def test_empty_index(self):
        """An empty collection should return zeros without calling get()."""
        col = Mock()
        col.name = "vault"
        col.count.return_value = 0

        with patch("index._get_collection", return_value=col):
            from index import get_status

            status = get_status()

        self.assertEqual(status["total_chunks"], 0)
        self.assertEqual(status["by_file_type"], {})
        col.get.assert_not_called()

    def test_unknown_file_type_defaults_to_unknown(self):
        """Chunks with no file_type key should be counted as 'unknown'."""
        col = Mock()
        col.name = "vault"
        col.count.return_value = 2
        col.get.return_value = {
            "metadatas": [
                {"file_path": "/a.md"},  # no file_type key
                {"file_path": "/b.md", "file_type": "text"},
            ]
        }

        with patch("index._get_collection", return_value=col):
            from index import get_status

            status = get_status()

        self.assertEqual(status["by_file_type"], {"unknown": 1, "text": 1})


class TestGetIndexedFilesBatching(unittest.TestCase):
    """index.get_indexed_files() must use paginated get() to avoid SQLite variable-limit errors."""

    def test_small_index_returns_deduplicated_paths(self):
        """Should return unique, sorted file paths."""
        col = Mock()
        col.count.return_value = 4
        col.get.return_value = {
            "metadatas": [
                {"file_path": "/vault/b.md", "file_type": "text"},
                {"file_path": "/vault/a.md", "file_type": "text"},
                {"file_path": "/vault/b.md", "file_type": "text"},  # duplicate
                {"file_path": "/vault/c.pdf", "file_type": "pdf"},
            ]
        }

        with patch("index._get_collection", return_value=col):
            from index import get_indexed_files

            paths = get_indexed_files()

        self.assertEqual(paths, ["/vault/a.md", "/vault/b.md", "/vault/c.pdf"])

    def test_batches_large_index(self):
        """Should iterate with limit/offset and accumulate across batches."""
        import index as idx_mod

        B = idx_mod._GET_BATCH_SIZE
        total_chunks = B + 5

        col = Mock()
        col.count.return_value = total_chunks

        # Batch 0: first B chunks → file0
        # Batch 1: last 5 chunks → file1
        get_calls = []

        def get_side_effect(**kwargs):
            get_calls.append((kwargs.get("limit"), kwargs.get("offset")))
            offset = kwargs.get("offset", 0)
            if offset == 0:
                return {"metadatas": [{"file_path": "/vault/file0.md", "file_type": "text"}] * B}
            else:
                return {"metadatas": [{"file_path": "/vault/file1.md", "file_type": "text"}] * 5}

        col.get.side_effect = get_side_effect

        with patch("index._get_collection", return_value=col):
            from index import get_indexed_files

            paths = get_indexed_files()

        self.assertEqual(paths, ["/vault/file0.md", "/vault/file1.md"])
        # Should have made calls with increasing offsets
        self.assertGreater(len(get_calls), 1)
        self.assertEqual(get_calls[0], (B, 0))
        self.assertEqual(get_calls[1], (B, B))

    def test_empty_index(self):
        """An empty collection returns empty list without get()."""
        col = Mock()
        col.count.return_value = 0

        with patch("index._get_collection", return_value=col):
            from index import get_indexed_files

            paths = get_indexed_files()

        self.assertEqual(paths, [])
        col.get.assert_not_called()

    def test_filters_empty_file_paths(self):
        """Chunks with empty file_path should be excluded."""
        col = Mock()
        col.count.return_value = 3
        col.get.return_value = {
            "metadatas": [
                {"file_path": "", "file_type": "text"},
                {"file_path": "/vault/real.md", "file_type": "text"},
                {"file_path": None, "file_type": "pdf"},
            ]
        }

        with patch("index._get_collection", return_value=col):
            from index import get_indexed_files

            paths = get_indexed_files()

        self.assertEqual(paths, ["/vault/real.md"])


if __name__ == "__main__":
    unittest.main()
