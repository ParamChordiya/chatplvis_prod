import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

def _best_device():
    # Prefer CUDA (NVIDIA), then Apple MPS, then CPU
    if torch.cuda.is_available():
        return "cuda"
    # PyTorch MPS (Apple Silicon). Guarded to avoid attribute errors on non-mac builds.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"

def compute_embeddings(compute_embeddings_csv,column = 'Function [CC]'):
    embeddings_file = 'data/embeddings_abs.npy' if column == 'Abstracts' else 'data/embeddings_cc.npy'
    device = _best_device()
    model = SentenceTransformer('sentence-transformers/gtr-t5-xl', device=device)

    # Use cached embeddings if present
    if os.path.exists(embeddings_file):
        return np.load(embeddings_file)

    # Compute fresh embeddings
    # df = pd.read_csv('data/mycobacterium_proteome_df.csv')
    df = pd.read_csv(compute_embeddings_csv)
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in dataframe.")
    texts = df[column].fillna('').astype(str).tolist()

    # Slightly larger batch on GPU for speed
    batch_size = 64 if device != "cpu" else 16

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    os.makedirs('data', exist_ok=True)
    np.save(embeddings_file, embeddings)
    return embeddings