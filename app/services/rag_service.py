"""
RAGService — orchestrates the full RAG pipeline: retrieval + generation.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Generator

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from app.api.schemas import ChatRequest
from app.core.config import Settings
from app.core.exceptions import DataError, EmbeddingError, LLMError, RateLimitedError
from app.services.data_service import DataService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.utils.cache import RAGCache
from app.utils.text import render_markdown_safe

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert computational biologist specializing in Mycobacterium proteomics. "
    "You have deep knowledge of M. tuberculosis biology, protein function, gene regulation, "
    "pathogenesis, and drug resistance mechanisms. "
    "You are given structured protein data from a proteome visualization tool. "
    "The context is split into SELECTED PROTEINS (explicitly chosen by the researcher) "
    "and RELATED PROTEINS (retrieved automatically by semantic similarity). "
    "Use the provided context AND your own expertise to answer comprehensively. "
    "Where relevant, note shared functions, biological relationships, and pathway memberships. "
    "Structure your response clearly with headers for multi-part questions."
)

# Count columns to include in context when present in the dataframe
_COUNT_COLS = ["Total_Counts", "Counts_1st", "Counts_2nd", "Counts_3rd", "Rank"]


class RAGService:
    """Orchestrates the full RAG pipeline: retrieval + generation."""

    def __init__(
        self,
        data_svc: DataService,
        embed_svc: EmbeddingService,
        llm_svc: LLMService,
        settings: Settings,
        cache: RAGCache,
    ) -> None:
        self._data_svc = data_svc
        self._embed_svc = embed_svc
        self._llm_svc = llm_svc
        self._settings = settings
        self._cache = cache

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _build_retrieval_query(
        self,
        message: str,
        node_ids: tuple[int, ...],
        df_indexed: pd.DataFrame,
        info_source: str,
    ) -> str:
        """
        Build a semantically rich retrieval query by combining the user's
        question with brief excerpts from the selected proteins' annotations.

        This anchors FAISS/BM25 to the biological domain of the selected
        proteins, rather than just matching the literal question text.
        """
        snippets: list[str] = []
        for nid in list(node_ids)[:5]:
            if nid in df_indexed.index:
                text = str(df_indexed.loc[nid].get(info_source, "")).strip()
                if text and text.lower() not in ("", "nan"):
                    snippets.append(text[:250])
        if snippets:
            return message + " " + " ".join(snippets)
        return message

    def _rrf_merge(
        self,
        faiss_indices: list[int],
        bm25_indices: list[int],
        k: int,
    ) -> list[int]:
        """Merge two ranked lists with Reciprocal Rank Fusion (1-based ranks)."""
        scores: dict[int, float] = {}
        for rank, idx in enumerate(faiss_indices, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        for rank, idx in enumerate(bm25_indices, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    def _hybrid_search(
        self,
        query: str,
        faiss_index: faiss.Index,
        bm25: BM25Okapi,
        df: pd.DataFrame,
        exclude_abs_ids: set[int],
    ) -> list[int]:
        """
        FAISS top-K + BM25 top-K → RRF merge → top-context_size row indices.
        Proteins already in the selected set are excluded from results.
        """
        df_len = len(df)
        cfg = self._settings

        # FAISS semantic search
        q_vec = self._embed_svc.encode_query(query).reshape(1, -1)
        n_search = min(cfg.faiss_top_k * 3, df_len)  # oversample to allow exclusion
        _distances, faiss_raw = faiss_index.search(q_vec, n_search)
        faiss_idxs = [
            int(i) for i in faiss_raw[0]
            if 0 <= i < df_len
            and int(df.iloc[i].get("abs_id", -1)) not in exclude_abs_ids
        ][:cfg.faiss_top_k]

        # BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = np.array(bm25.get_scores(tokenized_query), dtype=np.float64)
        # Exclude already-selected proteins from BM25 results
        abs_id_arr = df["abs_id"].to_numpy(dtype=int)
        excluded_mask = np.isin(abs_id_arr, list(exclude_abs_ids))
        bm25_scores[excluded_mask] = -1.0
        bm25_top_idxs = np.argsort(bm25_scores)[::-1][: cfg.bm25_top_k].tolist()

        merged = self._rrf_merge(faiss_idxs, bm25_top_idxs, k=cfg.rrf_k)
        return merged[: cfg.context_size]

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _rank_selected_by_relevance(
        self,
        node_ids: tuple[int, ...],
        embeddings: np.ndarray,
        df: pd.DataFrame,
        query: str,
    ) -> list[int]:
        """
        Rank the selected abs_ids by cosine similarity to the query embedding.
        Returns abs_ids sorted most-relevant first.

        Falls back to original order if any mapping step fails.
        """
        try:
            abs_id_to_pos = {int(v): i for i, v in enumerate(df["abs_id"])}
            positions = [abs_id_to_pos[nid] for nid in node_ids if nid in abs_id_to_pos]
            if not positions:
                return list(node_ids)

            sel_embeddings = embeddings[positions]            # (K, dim)
            q_vec = self._embed_svc.encode_query(query)      # (dim,) — already L2-normalized
            sims = sel_embeddings @ q_vec                     # cosine similarity

            ranked_positions = np.argsort(sims)[::-1]
            ordered_abs_ids = [
                int(df.iloc[positions[rp]]["abs_id"])
                for rp in ranked_positions
                if rp < len(positions)
            ]
            # Preserve any abs_ids that couldn't be mapped (put them at the end)
            mapped_set = set(ordered_abs_ids)
            extras = [nid for nid in node_ids if nid not in mapped_set]
            return ordered_abs_ids + extras

        except Exception:
            logger.debug("Relevance ranking failed, using original order.", exc_info=True)
            return list(node_ids)

    def _format_protein_row(
        self,
        row: pd.Series,
        info_source: str,
        label: str,
    ) -> str:
        """Format a single protein row into a readable context block."""
        protein_name = str(row.get("Protein names", "Unknown")).strip()
        gene_name    = str(row.get("Gene Names", "Unknown")).strip()
        organism     = str(row.get("Organism", "Unknown")).strip()
        info_text    = str(row.get(info_source, "")).strip()
        if info_text.lower() in ("", "nan"):
            info_text = "(no annotation)"

        lines = [
            f"[{label}]",
            f"Name:     {protein_name}",
            f"Gene:     {gene_name}",
            f"Organism: {organism}",
        ]

        # Include count/abundance data when available
        count_parts = []
        for col in _COUNT_COLS:
            val = row.get(col)
            if val is not None and str(val).lower() not in ("", "nan"):
                try:
                    count_parts.append(f"{col}: {float(val):.2f}")
                except (ValueError, TypeError):
                    pass
        if count_parts:
            lines.append("Counts:   " + " | ".join(count_parts))

        lines.append(f"Annotation ({info_source}): {info_text}")
        return "\n".join(lines)

    def _build_context(
        self,
        node_ids: tuple[int, ...],
        related_idxs: list[int],
        df: pd.DataFrame,
        df_indexed: pd.DataFrame,
        embeddings: np.ndarray,
        info_source: str,
        query: str,
    ) -> tuple[str, int, int]:
        """
        Build the structured context block for the LLM prompt.

        Selected proteins are ranked by relevance to the query; only the top
        max_selected_context are included with full detail. Related proteins
        from hybrid search follow.

        Returns (context_str, n_selected_shown, total_selected).
        """
        cfg = self._settings
        total_selected = len(node_ids)
        context_parts: list[str] = []
        seen_abs_ids: set[int] = set()
        char_budget = cfg.max_context_chars

        # --- Rank selected proteins by relevance to the query ---
        ranked_node_ids = self._rank_selected_by_relevance(
            node_ids, embeddings, df, query
        )

        selected_shown = 0
        for nid in ranked_node_ids[: cfg.max_selected_context]:
            if nid in df_indexed.index and nid not in seen_abs_ids:
                seen_abs_ids.add(nid)
                row = df_indexed.loc[nid]
                block = self._format_protein_row(row, info_source, "Selected Protein")
                if len("\n\n".join(context_parts + [block])) > char_budget:
                    break
                context_parts.append(block)
                selected_shown += 1

        # --- Related proteins from hybrid search ---
        related_shown = 0
        for row_idx in related_idxs:
            if row_idx < 0 or row_idx >= len(df):
                continue
            candidate_row = df.iloc[row_idx]
            candidate_abs_id = int(candidate_row.get("abs_id", -1))
            if candidate_abs_id in seen_abs_ids:
                continue
            seen_abs_ids.add(candidate_abs_id)
            block = self._format_protein_row(candidate_row, info_source, "Related Protein")
            if len("\n\n".join(context_parts + [block])) > char_budget:
                break
            context_parts.append(block)
            related_shown += 1

        logger.info(
            "Context: %d/%d selected proteins shown, %d related proteins added.",
            selected_shown, total_selected, related_shown,
        )

        context_str = "\n\n---\n\n".join(context_parts) if context_parts else "(No protein context available.)"
        return context_str, selected_shown, total_selected

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def answer(self, request: ChatRequest) -> str:
        """
        Full RAG pipeline.  Returns a sanitized HTML string.

        Steps:
          1.  Cache check.
          2.  Load CSV.
          3.  Get FAISS index + embeddings.
          4.  Get BM25 index.
          5.  Build a semantically rich retrieval query (message + protein snippets).
          6.  Hybrid search → top-K related row indices (excluding selected proteins).
          7.  Rank selected proteins by relevance to query; truncate to max_selected_context.
          8.  Assemble structured context block.
          9.  Assemble user prompt with selection summary header.
          10. LLM call.
          11. Markdown → sanitized HTML.
          12. Cache result.
        """
        csv_path = self._settings.csv_path

        # 1. Cache check
        cached = self._cache.get(
            request.message, request.info_source, csv_path, list(request.node_ids)
        )
        if cached is not None:
            logger.info("Response cache hit — returning early.")
            return cached

        try:
            # 2. Load CSV
            df = self._data_svc.load_proteome()

            if "abs_id" not in df.columns:
                return render_markdown_safe(
                    "**Error:** Dataset missing `abs_id` column. Please regenerate the CSV."
                )
            if request.info_source not in df.columns:
                return render_markdown_safe(
                    f"**Error:** Column `{request.info_source}` not found in dataset."
                )

            df_indexed = df.set_index("abs_id", drop=False)

            # 3. FAISS index + embeddings
            faiss_index, embeddings = self._embed_svc.get_or_build_index(request.info_source)

            if faiss_index.ntotal == 0:
                return render_markdown_safe(
                    "**Error:** FAISS index is empty — no proteins were indexed."
                )

            # 4. BM25 index
            bm25 = self._embed_svc.get_bm25(df, request.info_source)

            # 5. Build retrieval query — anchored to selected proteins' biology
            retrieval_query = self._build_retrieval_query(
                message=request.message,
                node_ids=request.node_ids,
                df_indexed=df_indexed,
                info_source=request.info_source,
            )

            # 6. Hybrid search (excludes already-selected proteins)
            related_row_idxs: list[int] = []
            if request.include_similar:
                try:
                    related_row_idxs = self._hybrid_search(
                        query=retrieval_query,
                        faiss_index=faiss_index,
                        bm25=bm25,
                        df=df,
                        exclude_abs_ids=set(request.node_ids),
                    )
                    logger.info("Hybrid search returned %d related proteins.", len(related_row_idxs))
                except Exception:
                    logger.exception("Hybrid search failed — proceeding without related proteins.")

            # 7–8. Build context (ranking + truncation happen inside)
            context_block, n_shown, n_total = self._build_context(
                node_ids=request.node_ids,
                related_idxs=related_row_idxs,
                df=df,
                df_indexed=df_indexed,
                embeddings=embeddings,
                info_source=request.info_source,
                query=request.message,
            )

            # 9. Assemble prompt with selection summary
            selection_note = (
                f"{n_total} proteins selected"
                if n_shown == n_total
                else f"{n_total} proteins selected — showing {n_shown} most relevant to your question"
            )
            related_note = (
                f"{len(related_row_idxs)} additional proteins retrieved by semantic similarity"
                if related_row_idxs
                else "no additional proteins retrieved"
            )

            user_prompt = (
                f"## Protein Context\n\n"
                f"Selection summary: {selection_note}; {related_note}.\n\n"
                f"{context_block}\n\n"
                f"---\n\n"
                f"## Question\n\n"
                f"{request.message}"
            )

            # 10. LLM call
            raw_text = self._llm_svc.complete(SYSTEM_PROMPT, user_prompt)

            # 11. Markdown → sanitized HTML
            html_response = render_markdown_safe(raw_text)

            # 12. Cache
            self._cache.set(
                request.message,
                request.info_source,
                csv_path,
                list(request.node_ids),
                html_response,
            )

            return html_response

        except RateLimitedError as exc:
            logger.warning("Rate limit: %s", exc)
            return render_markdown_safe(
                "**Rate limit reached.** Please wait a moment and try again."
            )
        except LLMError as exc:
            logger.error("LLM error: %s", exc)
            return render_markdown_safe(f"**AI service error:** {exc}")
        except (DataError, EmbeddingError) as exc:
            logger.error("%s: %s", type(exc).__name__, exc)
            return render_markdown_safe(f"**Data/embedding error:** {exc}")
        except Exception:
            logger.error("Unexpected error in RAGService.answer:\n%s", traceback.format_exc())
            return render_markdown_safe(
                "**An unexpected error occurred.** Please check server logs or try again."
            )

    # ------------------------------------------------------------------
    # Streaming entry point
    # ------------------------------------------------------------------

    def answer_stream(self, request: ChatRequest) -> Generator[str, None, None]:
        """
        Full RAG pipeline with streaming generation.

        Does all retrieval synchronously (fast, ~0.2 s with warm cache), then
        yields raw LLM text chunks as they arrive so the browser can render
        tokens progressively.  Raises ChatPLVisError subclasses on failure.
        """
        csv_path = self._settings.csv_path

        # 1. Load data
        df = self._data_svc.load_proteome()

        if "abs_id" not in df.columns:
            yield "**Error:** Dataset missing `abs_id` column."
            return
        if request.info_source not in df.columns:
            yield f"**Error:** Column `{request.info_source}` not found in dataset."
            return

        df_indexed = df.set_index("abs_id", drop=False)

        # 2. FAISS + BM25
        faiss_index, embeddings = self._embed_svc.get_or_build_index(request.info_source)
        bm25 = self._embed_svc.get_bm25(df, request.info_source)

        # 3. Retrieval (anchored to selected proteins' biology)
        retrieval_query = self._build_retrieval_query(
            message=request.message,
            node_ids=request.node_ids,
            df_indexed=df_indexed,
            info_source=request.info_source,
        )

        related_row_idxs: list[int] = []
        if request.include_similar:
            try:
                related_row_idxs = self._hybrid_search(
                    query=retrieval_query,
                    faiss_index=faiss_index,
                    bm25=bm25,
                    df=df,
                    exclude_abs_ids=set(request.node_ids),
                )
            except Exception:
                logger.exception("Hybrid search failed in answer_stream.")

        # 4. Context + prompt
        context_block, n_shown, n_total = self._build_context(
            node_ids=request.node_ids,
            related_idxs=related_row_idxs,
            df=df,
            df_indexed=df_indexed,
            embeddings=embeddings,
            info_source=request.info_source,
            query=request.message,
        )

        selection_note = (
            f"{n_total} proteins selected"
            if n_shown == n_total
            else f"{n_total} proteins selected — showing {n_shown} most relevant"
        )
        related_note = (
            f"{len(related_row_idxs)} additional proteins retrieved by semantic similarity"
            if related_row_idxs else "no additional proteins retrieved"
        )

        user_prompt = (
            f"## Protein Context\n\n"
            f"Selection summary: {selection_note}; {related_note}.\n\n"
            f"{context_block}\n\n"
            f"---\n\n"
            f"## Question\n\n"
            f"{request.message}"
        )

        # 5. Stream LLM tokens
        yield from self._llm_svc.complete_stream(SYSTEM_PROMPT, user_prompt)
