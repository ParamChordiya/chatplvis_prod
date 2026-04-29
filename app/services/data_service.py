"""
DataService — owns all CSV/DataFrame operations and plot data construction.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any

import numpy as np
import pandas as pd

from app.api.schemas import PlotState, ProteinNode
from app.core.config import Settings
from app.core.exceptions import DataError
from app.utils.text import clean_organism_name

logger = logging.getLogger(__name__)

_COLOR_PALETTE: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

_COUNTS_COLS: list[str] = [
    "Counts_1st",
    "Counts_2nd",
    "Counts_3rd",
    "Total_Counts",
    "Rank",
]


class DataService:
    """Owns all CSV/DataFrame operations and plot data construction."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # LRU DataFrame cache — keyed by (path, mtime_ns)
        self._df_cache: dict[tuple[str, int], pd.DataFrame] = {}
        self._df_cache_order: list[tuple[str, int]] = []
        self._df_cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lru_get(self, key: tuple[str, int]) -> pd.DataFrame | None:
        with self._df_cache_lock:
            if key in self._df_cache:
                self._df_cache_order.remove(key)
                self._df_cache_order.append(key)
                logger.debug("DataFrame cache hit for %s", key[0])
                return self._df_cache[key]
        return None

    def _lru_set(self, key: tuple[str, int], df: pd.DataFrame) -> None:
        with self._df_cache_lock:
            if key in self._df_cache:
                # Already stored by a racing thread — just promote
                self._df_cache_order.remove(key)
                self._df_cache_order.append(key)
                return
            if len(self._df_cache_order) >= self._settings.df_cache_maxsize:
                evict = self._df_cache_order.pop(0)
                self._df_cache.pop(evict, None)
                logger.debug("DataFrame cache evicted %s", evict[0])
            self._df_cache[key] = df
            self._df_cache_order.append(key)

    def _clear_cache(self) -> None:
        with self._df_cache_lock:
            self._df_cache.clear()
            self._df_cache_order.clear()

    def _assign_organism_colors(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Organism" not in df.columns:
            df["Color"] = _COLOR_PALETTE[0]
            return df
        df["Organism"] = df["Organism"].apply(clean_organism_name)
        organisms = df["Organism"].unique()
        color_map: dict[str, str] = {
            org: _COLOR_PALETTE[i % len(_COLOR_PALETTE)]
            for i, org in enumerate(organisms)
        }
        df["Color"] = df["Organism"].map(color_map)
        return df

    def _load_counts(self) -> pd.DataFrame:
        counts_path = self._settings.counts_path
        if not os.path.exists(counts_path):
            logger.info("Counts file not found, skipping: %s", counts_path)
            return pd.DataFrame()
        logger.info("Loading counts file: %s", counts_path)
        try:
            counts_df = pd.read_excel(counts_path)
        except Exception as exc:
            logger.warning("Could not read counts Excel file '%s': %s", counts_path, exc)
            return pd.DataFrame()
        counts_df["clean_orf"] = counts_df["orf"].apply(
            lambda x: re.sub(r"(?<=RV)BD", "", str(x), count=1)
        )
        counts_df["clean_name"] = counts_df["name"].apply(
            lambda x: re.sub(r"(?<=RV)BD", "", str(x), count=1)
        )
        return counts_df

    def _merge_counts_into_tb(
        self, tb_df: pd.DataFrame, counts_df: pd.DataFrame
    ) -> pd.DataFrame:
        if counts_df.empty or "Gene Names" not in tb_df.columns:
            return tb_df
        for _, row in counts_df.iterrows():
            pattern = (
                re.escape(str(row["clean_orf"]))
                + "|"
                + re.escape(str(row["clean_name"]))
            )
            mask = tb_df["Gene Names"].str.contains(
                pattern, case=False, na=False, regex=True
            )
            tb_df.loc[mask, _COUNTS_COLS] = [
                row["Counts_1st"],
                row["Counts_2nd"],
                row["Counts_3rd"],
                row["Total_Counts"],
                row["Rank"],
            ]
        return tb_df

    def _normalize_columns(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        for col in columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val == 0:
                df[col + "_normalized"] = 0.0
            else:
                df[col + "_normalized"] = (df[col] - min_val) / (max_val - min_val)
        return df

    def _build_plot_df(
        self,
        mycobacterium_df: pd.DataFrame,
        tb_df: pd.DataFrame,
        sel_col: str,
        sel_comp: str,
    ) -> pd.DataFrame:
        tb_df_subset = tb_df[["Entry", sel_col, "Rank", "abs_id"]].copy()

        if sel_comp == "All proteomes":
            plot_df = pd.merge(
                mycobacterium_df,
                tb_df_subset,
                how="left",
                on=["Entry", "abs_id"],
                suffixes=("", "_tb"),
            )
        elif sel_comp == "Mycobacterium tuberculosis":
            plot_df = tb_df.copy()
        else:
            # "vs X" — keep TB rows and the target organism
            other_org = sel_comp[3:]  # strip leading "vs "
            if "Organism" in mycobacterium_df.columns:
                mask = (
                    mycobacterium_df["Organism"] == "Mycobacterium tuberculosis"
                ) | mycobacterium_df["Organism"].str.contains(
                    other_org, case=False, na=False
                )
                subset = mycobacterium_df[mask]
            else:
                subset = mycobacterium_df
            plot_df = pd.merge(
                subset,
                tb_df_subset,
                how="left",
                on=["Entry", "abs_id"],
                suffixes=("", "_tb"),
            )

        plot_df[sel_col] = plot_df[sel_col].fillna(0)
        return plot_df

    def _build_nodes(
        self,
        plot_df: pd.DataFrame,
        sel_col: str,
    ) -> list[ProteinNode]:
        nodes: list[ProteinNode] = []
        for i in range(len(plot_df)):
            row = plot_df.iloc[i]
            protein: Any = row.get("Protein names", "N/A")
            organism: Any = row.get("Organism", "N/A")
            gene: Any = row.get("Gene Names", "N/A")
            pathway: Any = row.get("Pathway", "N/A")
            anot: Any = row.get("Annotation", "N/A")
            counts_val: Any = row.get(sel_col, "N/A")
            size: float = float(row.get("Size", 10))
            label_raw: Any = row.get("Cluster Label", "N/A")
            color: str = str(row.get("Color", _COLOR_PALETTE[0]))
            abs_id: int = int(row.get("abs_id", i))

            hover_text = (
                f"Protein Names: {protein}<br>"
                f"Organism: {organism}<br>"
                f"Gene Names: {gene}<br>"
                f"Pathway: {pathway}<br>"
                f"Counts: {counts_val}<br>"
                f"Annotation: {anot}<br>"
                f"Cluster: {label_raw}"
            )

            x = float(row.get("UMAP 1", 0.0)) if "UMAP 1" in row.index else 0.0
            y = float(row.get("UMAP 2", 0.0)) if "UMAP 2" in row.index else 0.0

            nodes.append(
                ProteinNode(
                    id=abs_id,
                    protein_name=str(protein),
                    label=hover_text,
                    x=x,
                    y=y,
                    group=str(label_raw),
                    size=size,
                    color=color,
                )
            )
        return nodes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_proteome(self) -> pd.DataFrame:
        """
        Load the proteome CSV from settings.csv_path.

        - Cached per (path, mtime_ns) — reloads only when file changes.
        - Adds abs_id column if missing (writes back to CSV, clears cache).
        - Applies organism name cleaning and color assignment.
        - Raises DataError on file-not-found or read failure.
        """
        csv_path = self._settings.csv_path

        if not os.path.exists(csv_path):
            raise DataError(f"Proteome CSV not found: {csv_path}")

        try:
            mtime_ns: int = os.stat(csv_path).st_mtime_ns
        except OSError as exc:
            raise DataError(f"Cannot stat CSV file '{csv_path}': {exc}") from exc

        cache_key = (csv_path, mtime_ns)
        cached_df = self._lru_get(cache_key)
        if cached_df is not None:
            return cached_df

        logger.info("Loading proteome CSV from disk: %s", csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            raise DataError(f"Failed to read CSV '{csv_path}': {exc}") from exc

        # Write-once abs_id injection
        if "abs_id" not in df.columns:
            logger.info("Adding abs_id column to CSV: %s", csv_path)
            df = df.copy()
            df["abs_id"] = range(len(df))
            try:
                df.to_csv(csv_path, index=False)
            except OSError as exc:
                raise DataError(
                    f"Could not write abs_id back to '{csv_path}': {exc}"
                ) from exc
            # Bust stale cache entries for this path
            self._clear_cache()
            try:
                mtime_ns = os.stat(csv_path).st_mtime_ns
            except OSError as exc:
                raise DataError(
                    f"Cannot re-stat CSV after writing abs_id: {exc}"
                ) from exc
            cache_key = (csv_path, mtime_ns)

        df = self._assign_organism_colors(df)
        self._lru_set(cache_key, df)
        return df

    def build_plot_data(
        self, state: PlotState
    ) -> tuple[list[ProteinNode], pd.DataFrame]:
        """
        Build the list of ProteinNode objects for the scatter plot.

        Returns (nodes, tb_df) — tb_df is needed by RAGService for context.
        Raises DataError on failure.
        """
        try:
            mycobacterium_df = self.load_proteome()

            # Isolate M. tuberculosis rows
            if "Organism" in mycobacterium_df.columns:
                tb_df = mycobacterium_df[
                    mycobacterium_df["Organism"] == "Mycobacterium tuberculosis"
                ].copy()
            else:
                tb_df = mycobacterium_df.copy()

            # Ensure count columns exist before merging
            for col in _COUNTS_COLS:
                if col not in tb_df.columns:
                    tb_df[col] = np.nan

            # Load and merge experimental counts
            counts_df = self._load_counts()
            tb_df = self._merge_counts_into_tb(tb_df, counts_df)

            # Min-max normalize count columns
            tb_df = self._normalize_columns(tb_df, _COUNTS_COLS)

            # Validate sel_col — fall back to first normalized column if missing
            sel_col = state.sel_col
            if sel_col not in tb_df.columns:
                fallback_candidates = [
                    c for c in tb_df.columns if c.endswith("_normalized")
                ]
                sel_col = (
                    fallback_candidates[0]
                    if fallback_candidates
                    else tb_df.columns[0]
                )
                logger.warning(
                    "sel_col %r not in tb_df columns; fell back to %r",
                    state.sel_col,
                    sel_col,
                )

            # Build the combined plot DataFrame
            plot_df = self._build_plot_df(
                mycobacterium_df, tb_df, sel_col, state.sel_comp
            )

            # Normalize sel_col → 0-1 regardless of whether it's already normalized
            col_vals = plot_df[sel_col].fillna(0).astype(float)
            c_min, c_max = col_vals.min(), col_vals.max()
            norm_vals = (col_vals - c_min) / (c_max - c_min) if c_max > c_min else col_vals * 0.0
            # Map to a 4-14 px visual size range; ensures low-count proteins stay visible
            plot_df["Size"] = (4 + norm_vals * 10).round(2)

            nodes = self._build_nodes(plot_df, sel_col)
            return nodes, tb_df

        except DataError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error in build_plot_data")
            raise DataError(f"Failed to build plot data: {exc}") from exc
