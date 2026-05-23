"""Tests for lexical and hybrid search (exact-term recall on specialist vocabulary).

Run with:  python -m pytest tests/test_lexical_search.py -v
Or plain:  python tests/test_lexical_search.py
"""

import json
import os
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCaseVariants(unittest.TestCase):
    """Case-variant generation for ChromaDB $contains queries."""

    def setUp(self):
        from index import _case_variants
        self._case_variants = _case_variants

    def test_mixed_case_term(self):
        variants = self._case_variants("mt-Keima")
        self.assertEqual(variants, ["mt-Keima", "mt-keima", "MT-KEIMA", "Mt-Keima"])

    def test_lowercase_term(self):
        variants = self._case_variants("autophagy")
        self.assertEqual(variants, ["autophagy", "AUTOPHAGY", "Autophagy"])

    def test_uppercase_term(self):
        variants = self._case_variants("PINK1")
        self.assertEqual(variants, ["PINK1", "pink1", "Pink1"])

    def test_cjk_no_variants(self):
        variants = self._case_variants("自噬")
        self.assertEqual(variants, ["自噬"])

    def test_no_duplicates(self):
        variants = self._case_variants("a")
        self.assertEqual(variants, ["a", "A"])


class TestExtractLexicalTerms(unittest.TestCase):
    """Query → search-term extraction with stop-word filtering."""

    def setUp(self):
        from index import _extract_lexical_terms
        self._extract = _extract_lexical_terms

    def test_mixed_cjk_english(self):
        terms = self._extract("mt-Keima autophagy 自噬")
        self.assertEqual(terms, ["mt-Keima", "autophagy", "自噬"])

    def test_stop_words_filtered(self):
        terms = self._extract("the mitochondria in cells")
        self.assertEqual(terms, ["mitochondria", "cells"])

    def test_single_char_filtered(self):
        terms = self._extract("a test")
        self.assertEqual(terms, ["test"])

    def test_cjk_always_kept(self):
        terms = self._extract("线粒体 自噬 的 研究")
        self.assertEqual(terms, ["线粒体", "自噬", "的", "研究"])

    def test_hyphenated_terms_preserved(self):
        terms = self._extract("mt-Keima LC3-II p62/SQSTM1")
        self.assertEqual(terms, ["mt-Keima", "LC3-II", "p62/SQSTM1"])

    def test_punctuation_stripped(self):
        terms = self._extract("mitophagy, autophagy; PINK1.")
        self.assertEqual(terms, ["mitophagy", "autophagy", "PINK1"])

    def test_empty_query(self):
        self.assertEqual(self._extract(""), [])

    def test_only_stop_words(self):
        self.assertEqual(self._extract("the a in of"), [])


class TestLexicalQuery(unittest.TestCase):
    """lexical_query() with mocked ChromaDB collection."""

    def setUp(self):
        import index as idx
        self.idx = idx
        self.mock_col = Mock()
        self.mock_col.count.return_value = 100
        patcher = patch.object(idx, "_get_collection", return_value=self.mock_col)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_empty_terms(self):
        results = self.idx.lexical_query([])
        self.assertEqual(results, [])

    def test_empty_collection(self):
        self.mock_col.count.return_value = 0
        results = self.idx.lexical_query(["mt-Keima"])
        self.assertEqual(results, [])

    def test_basic_single_term_match(self):
        """A single term should return matching chunks with score 0.80."""
        self.mock_col.get.return_value = {
            "ids": ["note.md::chunk_0"],
            "documents": ["mt-Keima is a fluorescent probe for mitophagy"],
            "metadatas": [{"file_path": "note.md", "chunk_id": "note.md::chunk_0", "file_type": "text"}],
        }
        results = self.idx.lexical_query(["mt-Keima"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["file_path"], "note.md")
        self.assertEqual(results[0]["score"], 0.80)
        self.assertEqual(results[0]["match_terms"], 1)
        self.assertIn("mt-Keima", results[0]["snippet"])

    def test_case_insensitive_matching(self):
        """Querying mt-keima (lowercase) should find mt-Keima via variant."""
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            variant = kwargs.get("where_document", {}).get("$contains", "")
            if variant in ("mt-keima",):
                return {
                    "ids": ["note.md::chunk_0"],
                    "documents": ["mt-Keima is a fluorescent probe"],
                    "metadatas": [{"file_path": "note.md", "chunk_id": "note.md::chunk_0", "file_type": "text"}],
                }
            return {"ids": [], "documents": [], "metadatas": []}

        self.mock_col.get.side_effect = side_effect
        results = self.idx.lexical_query(["mt-keima"])
        self.assertEqual(len(results), 1)
        self.assertIn("mt-Keima", results[0]["snippet"])
        # Only 1 term matched (not inflated by case variants)
        self.assertEqual(results[0]["match_terms"], 1)
        self.assertEqual(results[0]["score"], 0.80)

    def test_cjk_term_matching(self):
        """CJK terms (自噬) should match via substring."""
        self.mock_col.get.return_value = {
            "ids": ["note.md::chunk_0"],
            "documents": ["线粒体自噬是维持线粒体质量控制的关键过程"],
            "metadatas": [{"file_path": "note.md", "chunk_id": "note.md::chunk_0", "file_type": "text"}],
        }
        results = self.idx.lexical_query(["自噬"])
        self.assertEqual(len(results), 1)
        self.assertIn("自噬", results[0]["snippet"])

    def test_multi_term_scoring(self):
        """Chunks matching more terms should rank higher."""
        def side_effect(**kwargs):
            variant = kwargs.get("where_document", {}).get("$contains", "")
            if variant == "mitophagy":
                return {
                    "ids": ["chunk_a", "chunk_b"],
                    "documents": [
                        "mitophagy and autophagy are related processes",
                        "mitophagy is a selective form of autophagy",
                    ],
                    "metadatas": [
                        {"file_path": "a.md", "chunk_id": "chunk_a", "file_type": "text"},
                        {"file_path": "b.md", "chunk_id": "chunk_b", "file_type": "text"},
                    ],
                }
            elif variant == "autophagy":
                return {
                    "ids": ["chunk_a"],
                    "documents": ["mitophagy and autophagy are related processes"],
                    "metadatas": [{"file_path": "a.md", "chunk_id": "chunk_a", "file_type": "text"}],
                }
            return {"ids": [], "documents": [], "metadatas": []}

        self.mock_col.get.side_effect = side_effect
        results = self.idx.lexical_query(["mitophagy", "autophagy"])
        self.assertEqual(len(results), 2)

        # chunk_a (both terms) should be first
        self.assertEqual(results[0]["chunk_id"], "chunk_a")
        self.assertEqual(results[0]["match_terms"], 2)
        self.assertEqual(results[0]["score"], 0.85)  # 0.80 + 0.05 * 1

        # chunk_b (one term) second
        self.assertEqual(results[1]["chunk_id"], "chunk_b")
        self.assertEqual(results[1]["match_terms"], 1)
        self.assertEqual(results[1]["score"], 0.80)

    def test_file_type_filter(self):
        """Should only return chunks of matching file_type."""
        self.mock_col.get.return_value = {
            "ids": ["a.md::chunk_0", "b.pdf::chunk_0"],
            "documents": ["text note", "pdf content"],
            "metadatas": [
                {"file_path": "a.md", "chunk_id": "a.md::chunk_0", "file_type": "text"},
                {"file_path": "b.pdf", "chunk_id": "b.pdf::chunk_0", "file_type": "pdf"},
            ],
        }
        results = self.idx.lexical_query(["content"], file_type="text")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["file_type"], "text")

    def test_n_results_cap(self):
        """Should return at most n_results."""
        self.mock_col.get.return_value = {
            "ids": [f"c{i}" for i in range(20)],
            "documents": [f"doc {i}" for i in range(20)],
            "metadatas": [{"file_path": f"file_{i}.md", "chunk_id": f"c{i}", "file_type": "text"} for i in range(20)],
        }
        results = self.idx.lexical_query(["doc"], n_results=5)
        self.assertEqual(len(results), 5)

    def test_no_matches(self):
        """Should return empty list when no chunks match."""
        self.mock_col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        results = self.idx.lexical_query(["xyznonexistent"])
        self.assertEqual(results, [])


class TestHybridSearchMerge(unittest.TestCase):
    """Merging logic: lexical first, semantic deduped."""

    def setUp(self):
        import server
        self.server = server

    def test_hybrid_merge_lexical_first(self):
        """Lexical hits should appear before semantic hits in merged output."""
        with patch.object(self.server, "embed_text", return_value=[0.1] * 1536), \
             patch.object(self.server.index, "query", return_value=[
                 {"file_path": "a.md", "chunk_id": "a::0", "file_type": "text", "score": 0.75, "snippet": "semantic a"},
                 {"file_path": "b.md", "chunk_id": "b::0", "file_type": "text", "score": 0.70, "snippet": "semantic b"},
             ]), \
             patch.object(self.server, "_extract_lexical_terms", return_value=["mt-Keima"]), \
             patch.object(self.server, "_lexical_query", return_value=[
                 {"file_path": "b.md", "chunk_id": "b::0", "file_type": "text", "score": 0.85, "snippet": "lexical b match", "match_terms": 1},
                 {"file_path": "c.md", "chunk_id": "c::0", "file_type": "text", "score": 0.80, "snippet": "lexical c match", "match_terms": 1},
             ]):
            result_json = self.server.vault_hybrid_search("mt-Keima autophagy", n_results=10)
            results = json.loads(result_json)

            sources = [r["source"] for r in results]
            self.assertEqual(sources[:2], ["lexical", "lexical"])  # first two are lexical
            self.assertEqual(sources[2], "semantic")

            chunk_ids = [r["chunk_id"] for r in results]
            self.assertIn("b::0", chunk_ids)
            self.assertIn("c::0", chunk_ids)
            self.assertIn("a::0", chunk_ids)

    def test_hybrid_no_lexical_terms(self):
        """When query has only stop-words, lexical is skipped."""
        with patch.object(self.server, "embed_text", return_value=[0.1] * 1536), \
             patch.object(self.server.index, "query", return_value=[
                 {"file_path": "x.md", "chunk_id": "x::0", "file_type": "text", "score": 0.90, "snippet": "result"}
             ]), \
             patch.object(self.server, "_extract_lexical_terms", return_value=[]):
            result_json = self.server.vault_hybrid_search("the a in", n_results=5)
            results = json.loads(result_json)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["source"], "semantic")

    def test_hybrid_deduplication(self):
        """A chunk appearing in both lexical and semantic should only appear once."""
        shared_chunk = {"file_path": "shared.md", "chunk_id": "shared::0", "file_type": "text", "score": 0.95, "snippet": "shared content", "match_terms": 2}

        with patch.object(self.server, "embed_text", return_value=[0.1] * 1536), \
             patch.object(self.server.index, "query", return_value=[shared_chunk.copy()]), \
             patch.object(self.server, "_extract_lexical_terms", return_value=["shared"]), \
             patch.object(self.server, "_lexical_query", return_value=[shared_chunk]):
            result_json = self.server.vault_hybrid_search("shared", n_results=10)
            results = json.loads(result_json)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["source"], "lexical")


class TestLexicalSearchTool(unittest.TestCase):
    """vault_lexical_search MCP tool integration tests."""

    def setUp(self):
        import server
        self.server = server

    def test_basic_call(self):
        with patch.object(self.server, "_extract_lexical_terms", return_value=["mt-Keima"]), \
             patch.object(self.server, "_lexical_query", return_value=[
                 {"file_path": "note.md", "chunk_id": "n::0", "file_type": "text", "score": 0.80, "snippet": "mt-Keima assay", "match_terms": 1}
             ]):
            result_json = self.server.vault_lexical_search("mt-Keima")
            results = json.loads(result_json)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["file_path"], "note.md")
            self.assertIn("match_terms", results[0])

    def test_invalid_file_type(self):
        result_json = self.server.vault_lexical_search("test", file_type="invalid")
        results = json.loads(result_json)
        self.assertIn("error", results)

    def test_n_results_clamped(self):
        """n_results > 50 should be clamped to 50."""
        with patch.object(self.server, "_extract_lexical_terms", return_value=["test"]), \
             patch.object(self.server, "_lexical_query") as mock_lq:
            self.server.vault_lexical_search("test", n_results=100)
            call_args = mock_lq.call_args
            self.assertEqual(call_args[1]["n_results"], 50)


class TestScoreNormalization(unittest.TestCase):
    """Lexical scores should be bounded 0.80–0.99."""

    def test_single_term_score(self):
        """One matching term → base 0.80 (not inflated by case variants)."""
        import index as idx
        mock_col = Mock()
        mock_col.count.return_value = 100
        # Side effect: four variants (mt-Keima, mt-keima, MT-KEIMA, Mt-Keima)
        # but only the first returns a match — score should be 0.80, not 0.95
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            variant = kwargs.get("where_document", {}).get("$contains", "")
            if variant == "mt-Keima":  # exact match only
                return {
                    "ids": ["n::0"],
                    "documents": ["mt-Keima assay"],
                    "metadatas": [{"file_path": "n.md", "chunk_id": "n::0", "file_type": "text"}],
                }
            return {"ids": [], "documents": [], "metadatas": []}

        mock_col.get.side_effect = side_effect
        with patch.object(idx, "_get_collection", return_value=mock_col):
            results = idx.lexical_query(["mt-Keima"])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["match_terms"], 1)
            self.assertEqual(results[0]["score"], 0.80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
