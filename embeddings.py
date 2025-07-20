import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ────────────────────── Configurable variables ──────────────────────
DATA_DIR:          Path = Path("data")
CSV_FILE:          Path = DATA_DIR / "mtuberculosis_df_abs.csv"
MODEL_NAME:        str  = "sentence-transformers/gtr-t5-xl"
DEVICE:            str  = "cpu"                       # set "cuda" if GPU is available
DEFAULT_COLUMN:    str  = "Function [CC]"

# Map column names ➜ filename suffixes (add more here as needed)
COLUMN_TO_SUFFIX: dict = {
    "Abstracts":      "abs",
    "Function [CC]":  "cc",
}

# Derived template for cached embedding files
EMBED_TEMPLATE: Path = DATA_DIR / "embeddings_{suffix}.npy"
# ────────────────────────────────────────────────────────────────────

# Initialise model once (avoids reloading on every call)
_model = SentenceTransformer(MODEL_NAME, device=DEVICE)


def compute_embeddings(column):
    """
    Return cached embeddings for `column`, or compute & cache them if missing.

    Args:
        column: Name of the dataframe column whose text will be embedded.

    Returns:
        embeddings (np.ndarray): shape = (num_rows, embedding_dim)
    """
    suffix = COLUMN_TO_SUFFIX.get(column, column.lower().replace(" ", "_"))
    emb_path = Path(str(EMBED_TEMPLATE).format(suffix=suffix))

    if emb_path.exists():
        return np.load(emb_path)

    df = pd.read_csv(CSV_FILE)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {CSV_FILE.name}")

    texts = df[column].fillna("").tolist()
    embeddings = _model.encode(texts, show_progress_bar=True)

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    return embeddings
