"""
EmbeddingService — owns the SentenceTransformer model, FAISS index,
BM25 index, and query embedding LRU cache.
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
import time

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Owns the embedding model, FAISS index, BM25 index, and query cache."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # Singleton model — loaded lazily
        self._model: SentenceTransformer | None = None
        self._model_lock = threading.Lock()

        # LRU query-embedding cache
        self._query_cache: dict[str, np.ndarray] = {}
        self._query_cache_order: list[str] = []
        self._query_cache_lock = threading.Lock()

        # BM25 in-memory cache — keyed by (column, csv_mtime)
        self._bm25_cache: dict[tuple[str, float], BM25Okapi] = {}
        self._bm25_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model singleton
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        """Return the singleton SentenceTransformer, loading it on first call."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            device = "cuda"
                        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                            device = "mps"   # Apple Silicon GPU — faster and avoids CPU thread bugs
                        else:
                            device = "cpu"
                    except ImportError:
                        device = "cpu"

                    logger.info(
                        "Loading SentenceTransformer '%s' on device=%s (first load)",
                        self._settings.embedding_model,
                        device,
                    )
                    t0 = time.monotonic()
                    try:
                        self._model = SentenceTransformer(
                            self._settings.embedding_model, device=device
                        )
                    except Exception as exc:
                        raise EmbeddingError(
                            f"Failed to load SentenceTransformer "
                            f"'{self._settings.embedding_model}': {exc}"
                        ) from exc
                    elapsed = time.monotonic() - t0
                    logger.info(
                        "SentenceTransformer loaded in %.1fs on device=%s.",
                        elapsed,
                        device,
                    )
        return self._model  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # File-path helpers
    # ------------------------------------------------------------------

    def _col_slug(self, column: str) -> str:
        return "_abs" if "abstract" in column.lower() else "_cc"

    def _index_paths(
        self, csv_path: str, column: str
    ) -> tuple[str, str, str]:
        """Return (index_file, npy_file, meta_file) for a given CSV + column."""
        base = os.path.splitext(os.path.basename(csv_path))[0]
        folder = os.path.dirname(csv_path) or "."
        slug = self._col_slug(column)
        index_file = os.path.join(folder, f"{base}_embeddings{slug}.index")
        npy_file = os.path.join(folder, f"{base}_embeddings{slug}.npy")
        meta_file = os.path.join(folder, f"{base}_embeddings{slug}_meta.pkl")
        return index_file, npy_file, meta_file

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------

    def get_or_build_index(
        self, column: str
    ) -> tuple[faiss.Index, np.ndarray]:
        """
        Return (faiss_index, embeddings) for settings.csv_path and column.

        Cache is valid when meta.num_rows == len(df) AND
        |meta.csv_mtime - current| < 1e-3.

        On miss: encode all texts with batch_size=64, L2-normalize,
        build IndexFlatIP, persist .index + .npy + .pkl.

        Raises EmbeddingError on any failure.
        """
        csv_path = self._settings.csv_path

        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")

            df = pd.read_csv(csv_path)
            if column not in df.columns:
                raise ValueError(
                    f"Column '{column}' not found in CSV '{csv_path}'"
                )

            index_file, npy_file, meta_file = self._index_paths(csv_path, column)
            csv_mtime = os.path.getmtime(csv_path)
            num_rows = len(df)

            # ----------------------------------------------------------
            # Attempt cache load
            # ----------------------------------------------------------
            all_cache_files = [index_file, npy_file, meta_file]
            if all(os.path.exists(p) for p in all_cache_files):
                try:
                    with open(meta_file, "rb") as fh:
                        meta: dict = pickle.load(fh)

                    cache_valid = (
                        meta.get("num_rows") == num_rows
                        and abs(meta.get("csv_mtime", 0) - csv_mtime) < 1e-3
                    )

                    if cache_valid:
                        logger.info(
                            "Loading FAISS index from cache for column='%s'",
                            column,
                        )
                        index = faiss.read_index(index_file)
                        embeddings = np.load(npy_file)
                        logger.info(
                            "Cache hit: %d vectors, dim=%d",
                            embeddings.shape[0],
                            embeddings.shape[1],
                        )
                        return index, embeddings
                    else:
                        logger.info(
                            "Cache invalid (mtime or row count changed). "
                            "Rebuilding for column='%s'.",
                            column,
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to load FAISS cache (%s). Rebuilding.", exc
                    )

            # ----------------------------------------------------------
            # Build from scratch
            # ----------------------------------------------------------
            logger.info(
                "Building FAISS index for column='%s', %d rows…",
                column,
                num_rows,
            )
            texts = df[column].fillna("").tolist()
            model = self._get_model()

            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                num_workers=0,        # no forking — required on macOS
            ).astype(np.float32)

            # L2-normalize (cosine similarity via inner product)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

            # Persist
            faiss.write_index(index, index_file)
            np.save(npy_file, embeddings)
            meta_out = {"num_rows": num_rows, "dim": dim, "csv_mtime": csv_mtime}
            with open(meta_file, "wb") as fh:
                pickle.dump(meta_out, fh)

            logger.info(
                "FAISS index built and saved. vectors=%d, dim=%d, "
                "files: %s, %s, %s",
                num_rows,
                dim,
                index_file,
                npy_file,
                meta_file,
            )
            return index, embeddings

        except EmbeddingError:
            raise
        except Exception as exc:
            logger.exception("get_or_build_index failed for column='%s'", column)
            raise EmbeddingError(
                f"Failed to build/load FAISS index for column '{column}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        """
        Encode a single query string, L2-normalize it, and return a float32 vector.
        Results are LRU-cached (max settings.query_cache_maxsize).
        """
        with self._query_cache_lock:
            if text in self._query_cache:
                self._query_cache_order.remove(text)
                self._query_cache_order.append(text)
                return self._query_cache[text].copy()

        # Encode outside the lock — this is the slow part
        try:
            model = self._get_model()
            vec = model.encode(
                [text], batch_size=1, show_progress_bar=False
            )[0].astype(np.float32)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Query encoding failed: {exc}") from exc

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        with self._query_cache_lock:
            # Double-check: another thread may have populated while we encoded
            if text not in self._query_cache:
                maxsize = self._settings.query_cache_maxsize
                if len(self._query_cache_order) >= maxsize:
                    evict_key = self._query_cache_order.pop(0)
                    self._query_cache.pop(evict_key, None)
                self._query_cache[text] = vec
                self._query_cache_order.append(text)

        return vec.copy()

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------

    def get_bm25(self, df: pd.DataFrame, column: str) -> BM25Okapi:
        """
        Return a BM25Okapi index for df[column].

        Cached in-memory keyed by (column, csv_mtime).
        Tokenization: doc.lower().split()
        """
        csv_path = self._settings.csv_path
        try:
            csv_mtime = os.path.getmtime(csv_path)
        except OSError:
            csv_mtime = 0.0

        cache_key = (column, csv_mtime)

        with self._bm25_lock:
            if cache_key in self._bm25_cache:
                logger.debug("BM25 cache hit for column='%s'", column)
                return self._bm25_cache[cache_key]

        logger.info(
            "Building BM25 index for column='%s' (%d docs)…", column, len(df)
        )
        try:
            corpus = df[column].fillna("").tolist()
            # BM25Okapi divides by average doc length — empty lists cause ZeroDivisionError
            tokenized = [doc.lower().split() or ["_"] for doc in corpus]
            bm25 = BM25Okapi(tokenized)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to build BM25 index for column '{column}': {exc}"
            ) from exc

        with self._bm25_lock:
            self._bm25_cache[cache_key] = bm25

        logger.info("BM25 index built for column='%s'.", column)
        return bm25
