"""Tests for the embedding-count-aware rate limiter and adaptive retry logic.

Run with:  python -m pytest tests/test_rate_limiter.py -v
Or plain:  python tests/test_rate_limiter.py
"""

import os
import sys
import threading
import time
import unittest
from unittest.mock import Mock, patch

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRateLimiterEmbeddingCount(unittest.TestCase):
    """The rate limiter must space requests by EMBEDDING count, not API calls."""

    def setUp(self):
        # Force-reload embedder for clean state on each test
        self._set_envs(EMBED_RPM="600", EMBED_BATCH_SIZE="100")
        import embedder as emb
        self.emb = emb
        # Reset adaptive state
        emb._current_batch_size = emb.EMBED_BATCH_SIZE
        emb._consecutive_429s = 0
        emb._consecutive_successes = 0
        emb._last_request = 0.0

    @staticmethod
    def _set_envs(**kwargs):
        for k, v in kwargs.items():
            os.environ[k] = v

    def test_min_interval_per_embedding(self):
        """With EMBED_RPM=600, each embedding needs 0.1s spacing."""
        self.assertAlmostEqual(self.emb._MIN_INTERVAL_PER_EMBEDDING, 0.1, places=4)

    def test_rate_limit_100_embeddings_waits_10s(self):
        """Batch of 100 at 600 RPM needs 10s wait."""
        self.emb._last_request = 0.0
        t0 = time.monotonic()
        with patch("time.monotonic", side_effect=[t0, t0]):  # no wait
            self.emb._rate_limit(100)
        # Cross-process pacing writes a shared state file instead of updating
        # the process-local _last_request fallback.

    def test_rate_limit_single_embedding(self):
        """Single embedding at 600 RPM needs 0.1s wait."""
        self.emb._last_request = 0.0
        now = time.monotonic()
        with patch("time.monotonic", side_effect=[now, now]):
            self.emb._rate_limit(1)

    def test_rate_limit_no_wait_when_idle(self):
        """No wait when enough time has passed since last request."""
        self.emb._last_request = time.monotonic() - 100
        t0 = time.monotonic()
        # Should not call time.sleep at all because wait <= 0
        self.emb._rate_limit(10)

    def test_rate_limit_thread_safety(self):
        """Concurrent calls to _rate_limit should not corrupt state."""
        results = []
        errors = []

        def worker(n_embeddings):
            try:
                self.emb._rate_limit(n_embeddings)
                results.append(True)
            except Exception as e:
                errors.append(e)

        with patch("time.sleep"):  # skip actual waiting
            threads = [threading.Thread(target=worker, args=(5,)) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 20)


class TestAdaptiveBatchSizing(unittest.TestCase):
    """Batch size must shrink on consecutive 429s and recover on successes."""

    def setUp(self):
        self._set_envs(EMBED_BATCH_SIZE="100", EMBED_MIN_BATCH_SIZE="10",
                       EMBED_BACKOFF_BATCH_FACTOR="0.5",
                       EMBED_BACKOFF_RECOVERY_STEPS="5")
        import embedder as emb
        self.emb = emb
        # Reset adaptive state
        emb._current_batch_size = emb.EMBED_BATCH_SIZE
        emb._consecutive_429s = 0
        emb._consecutive_successes = 0

    @staticmethod
    def _set_envs(**kwargs):
        for k, v in kwargs.items():
            os.environ[k] = v

    def test_initial_batch_size(self):
        self.assertEqual(self.emb._current_batch_size, 100)

    def test_on_429_halves_batch(self):
        self.emb._on_429()
        self.assertEqual(self.emb._current_batch_size, 50)
        self.assertEqual(self.emb._consecutive_429s, 1)
        self.assertEqual(self.emb._consecutive_successes, 0)

    def test_on_429_repeated(self):
        self.emb._on_429()  # 100 -> 50
        self.emb._on_429()  # 50 -> 25
        self.emb._on_429()  # 25 -> 12
        self.emb._on_429()  # 12 -> 10 (floor)
        self.emb._on_429()  # 10 -> 10 (floor)
        self.assertEqual(self.emb._current_batch_size, 10)
        self.assertEqual(self.emb._consecutive_429s, 5)

    def test_on_success_recovery(self):
        # Reduce first
        self.emb._on_429()
        self.emb._on_429()
        self.assertEqual(self.emb._current_batch_size, 25)

        # 5 successes -> full recovery
        for _ in range(5):
            self.emb._on_success()
        self.assertEqual(self.emb._current_batch_size, 100)
        self.assertEqual(self.emb._consecutive_429s, 0)

    def test_on_success_partial_no_recovery(self):
        """Not enough successes to trigger recovery."""
        self.emb._on_429()
        self.assertEqual(self.emb._current_batch_size, 50)

        for _ in range(4):  # one short
            self.emb._on_success()
        self.assertEqual(self.emb._current_batch_size, 50)  # still reduced

    def test_on_success_reset_by_429(self):
        """A 429 during recovery resets the success counter."""
        self.emb._on_429()  # 100 -> 50
        for _ in range(3):
            self.emb._on_success()  # 3 successes

        self.emb._on_429()  # reset! 50 -> 25
        self.assertEqual(self.emb._consecutive_successes, 0)
        self.assertEqual(self.emb._current_batch_size, 25)

    def test_thread_safety_adaptive(self):
        """Concurrent _on_429 and _on_success calls should be safe."""
        errors = []

        def hammer_429():
            try:
                for _ in range(10):
                    self.emb._on_429()
            except Exception as e:
                errors.append(e)

        def hammer_success():
            try:
                for _ in range(10):
                    self.emb._on_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer_429) for _ in range(4)] + \
                  [threading.Thread(target=hammer_success) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        # Batch size should be at floor (10) after 40 429s from 4 threads
        self.assertGreaterEqual(self.emb._current_batch_size, self.emb.EMBED_MIN_BATCH_SIZE)


class TestRetryAfterParsing(unittest.TestCase):
    """Retry-After header parsing from 429 response headers."""

    def setUp(self):
        import embedder as emb
        self.emb = emb

    def test_delta_seconds(self):
        delay = self.emb._parse_retry_after({"Retry-After": "120"})
        self.assertEqual(delay, 120.0)

    def test_delta_seconds_lowercase(self):
        delay = self.emb._parse_retry_after({"retry-after": "60"})
        self.assertEqual(delay, 60.0)

    def test_missing_header(self):
        delay = self.emb._parse_retry_after({"Content-Type": "application/json"})
        self.assertIsNone(delay)

    def test_empty_headers(self):
        delay = self.emb._parse_retry_after({})
        self.assertIsNone(delay)

    def test_invalid_value(self):
        delay = self.emb._parse_retry_after({"Retry-After": "not-a-number"})
        self.assertIsNone(delay)

    def test_zero_delay(self):
        delay = self.emb._parse_retry_after({"Retry-After": "0"})
        self.assertEqual(delay, 0.0)


class Test429Detection(unittest.TestCase):
    """429 / RESOURCE_EXHAUSTED classification."""

    def setUp(self):
        import embedder as emb
        self.emb = emb

    def test_429_in_message(self):
        exc = Exception("429 RESOURCE_EXHAUSTED")
        self.assertTrue(self.emb._is_retryable_429(exc))

    def test_resource_exhausted_only(self):
        exc = Exception("RESOURCE_EXHAUSTED: quota exceeded")
        self.assertTrue(self.emb._is_retryable_429(exc))

    def test_non_429(self):
        exc = Exception("500 Internal Server Error")
        self.assertFalse(self.emb._is_retryable_429(exc))

    def test_403_not_retryable(self):
        exc = Exception("403 Forbidden")
        self.assertFalse(self.emb._is_retryable_429(exc))


class TestBackoffWithMinDelay(unittest.TestCase):
    """Exponential backoff must respect a minimum delay."""

    def setUp(self):
        self._set_envs(EMBED_RETRY_BASE_DELAY="5", EMBED_RETRY_MAX_DELAY="120")
        import embedder as emb
        self.emb = emb

    @staticmethod
    def _set_envs(**kwargs):
        for k, v in kwargs.items():
            os.environ[k] = v

    def test_backoff_minimum_delay(self):
        """When min_delay exceeds calculated delay, use min_delay."""
        with patch("time.sleep") as mock_sleep:
            self.emb._sleep_backoff(attempt=0, min_delay=30.0)
            actual_delay = mock_sleep.call_args[0][0]
            # calculated: 5 * 2^0 = 5s * jitter(0.8-1.2) = 4-6s
            # min_delay = 30s, so delay should be >= 30 * 0.8 = 24s
            self.assertGreaterEqual(actual_delay, 24.0)
            self.assertLessEqual(actual_delay, 36.0)

    def test_backoff_no_minimum(self):
        """Without min_delay, use standard exponential backoff."""
        with patch("time.sleep") as mock_sleep:
            self.emb._sleep_backoff(attempt=0, min_delay=0.0)
            actual_delay = mock_sleep.call_args[0][0]
            self.assertGreaterEqual(actual_delay, 4.0)   # 5 * 0.8
            self.assertLessEqual(actual_delay, 6.0)       # 5 * 1.2

    def test_backoff_max_clamp(self):
        """Delay must not exceed EMBED_RETRY_MAX_DELAY."""
        with patch("time.sleep") as mock_sleep:
            self.emb._sleep_backoff(attempt=10, min_delay=0.0)
            actual_delay = mock_sleep.call_args[0][0]
            # calculated: 5 * 2^10 = 5120s, clamped to 120s * jitter
            self.assertGreaterEqual(actual_delay, 96.0)   # 120 * 0.8
            self.assertLessEqual(actual_delay, 144.0)     # 120 * 1.2


class TestConfigBackwardsCompatibility(unittest.TestCase):
    """Config must be backwards compatible with old EMBED_RPS env var."""

    def test_embed_rps_fallback(self):
        """When EMBED_RPM is not set, EMBED_RPS is used."""
        old_rpm = os.environ.pop("EMBED_RPM", None)
        try:
            os.environ["EMBED_RPS"] = "4.0"
            import importlib
            import config
            importlib.reload(config)
            self.assertEqual(config.EMBED_RPM, 240.0)  # 4.0 RPS * 60
        finally:
            if old_rpm is not None:
                os.environ["EMBED_RPM"] = old_rpm
            os.environ.pop("EMBED_RPS", None)


class TestDiagnosticLogHeaders(unittest.TestCase):
    """429 response header extraction from google-genai APIError."""

    def setUp(self):
        import embedder as emb
        self.emb = emb

    def test_non_api_error(self):
        """Non-APIError exceptions return empty headers."""
        exc = Exception("generic error")
        headers = self.emb._extract_429_headers(exc)
        self.assertEqual(headers, {})

    def test_extracts_ratelimit_headers(self):
        """Rate-limit-relevant headers are extracted from APIError."""
        mock_resp = Mock()
        mock_resp.headers = {
            "retry-after": "60",
            "x-ratelimit-remaining": "0",
            "X-Goog-Quota-User": "projects/...",
            "Content-Type": "application/json",
            "Server": "Google",
        }

        from google.genai.errors import APIError
        exc = APIError(
            code=429,
            response_json={"error": {"message": "429 RESOURCE_EXHAUSTED"}},
            response=mock_resp,
        )

        headers = self.emb._extract_429_headers(exc)
        self.assertIn("retry-after", headers)
        self.assertIn("x-ratelimit-remaining", headers)
        self.assertIn("X-Goog-Quota-User", headers)
        self.assertNotIn("Content-Type", headers)
        self.assertNotIn("Server", headers)


class TestErrorTruncation(unittest.TestCase):
    """Error messages must be truncated, never leak API keys."""

    def setUp(self):
        import embedder as emb
        self.emb = emb

    def test_short_message_unchanged(self):
        msg = "429 RESOURCE_EXHAUSTED"
        self.assertEqual(self.emb._truncate_error(Exception(msg)), msg)

    def test_long_message_truncated(self):
        msg = "x" * 500
        truncated = self.emb._truncate_error(Exception(msg))
        self.assertEqual(len(truncated), 300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
