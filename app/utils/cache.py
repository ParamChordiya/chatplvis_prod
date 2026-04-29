"""
Thread-safe in-memory response cache for the RAG pipeline.

Features:
  - Keyed by (message, info_source, csv_path, node_ids) — all inputs hashed to a
    stable SHA-256 hex string so no mutable types escape the cache boundary.
  - TTL expiry: entries older than `ttl_seconds` are silently dropped on access.
  - LRU eviction: when `maxsize` is reached the oldest entry is removed.
  - All operations protected by a single reentrant lock.
"""

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)


class RAGCache:
    """LRU cache with per-entry TTL for chatbot responses."""

    def __init__(self, maxsize: int = 128, ttl_seconds: float = 3600.0) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")

        self._maxsize = maxsize
        self._ttl = ttl_seconds
        # OrderedDict maps key -> (response_html, insertion_monotonic_timestamp)
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(
        self,
        message: str,
        info_source: str,
        csv_path: str,
        node_ids: list[int],
    ) -> str:
        """
        Hash all inputs into a stable, fixed-length cache key.

        node_ids is sorted so that [1, 2] and [2, 1] produce the same key.
        csv_path is resolved to an absolute path for consistency across callers.
        """
        try:
            resolved_path = os.path.abspath(csv_path)
        except Exception:
            resolved_path = csv_path

        payload = json.dumps(
            {
                "message": message,
                "info_source": info_source,
                "csv_path": resolved_path,
                "node_ids": sorted(node_ids),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        return (time.monotonic() - timestamp) > self._ttl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        message: str,
        info_source: str,
        csv_path: str,
        node_ids: list[int],
    ) -> str | None:
        """
        Return cached response string, or None on a miss (including TTL expiry).
        On a cache hit the entry is promoted to most-recently-used.
        """
        key = self._make_key(message, info_source, csv_path, node_ids)
        with self._lock:
            if key not in self._store:
                logger.debug("RAGCache miss  key=%.8s...", key)
                return None

            value, ts = self._store[key]
            if self._is_expired(ts):
                logger.debug("RAGCache TTL expired  key=%.8s...", key)
                del self._store[key]
                return None

            # Promote to MRU position
            self._store.move_to_end(key)
            logger.debug("RAGCache hit  key=%.8s...", key)
            return value

    def set(
        self,
        message: str,
        info_source: str,
        csv_path: str,
        node_ids: list[int],
        value: str,
    ) -> None:
        """
        Store a response.  Re-inserting an existing key resets its TTL and
        promotes it to MRU.  Evicts the LRU entry when maxsize is exceeded.
        """
        key = self._make_key(message, info_source, csv_path, node_ids)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            else:
                if len(self._store) >= self._maxsize:
                    evicted_key, _ = self._store.popitem(last=False)
                    logger.debug("RAGCache evicted LRU  key=%.8s...", evicted_key)
            self._store[key] = (value, time.monotonic())
            logger.debug(
                "RAGCache stored  key=%.8s...  store_size=%d", key, len(self._store)
            )

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()
        logger.info("RAGCache cleared.")

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"RAGCache(maxsize={self._maxsize}, ttl={self._ttl}s, "
                f"current_size={len(self._store)})"
            )
