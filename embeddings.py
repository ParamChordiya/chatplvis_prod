# embeddings.py
import os
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def get_faiss_index_file(csv_path, column):
    """Compute the FAISS index filename based on CSV path & column."""
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    folder = os.path.dirname(csv_path)

    if 'abstract' in column.lower():
        return os.path.join(folder, f"{base_name}_embeddings_abs.index")
    else:
        return os.path.join(folder, f"{base_name}_embeddings_cc.index")

def get_faiss_meta_file(csv_path, column):
    """
    We also store a small meta file (pickle) to record:
      - number of rows
      - dimension of embeddings
    So we can verify if the index matches our CSV.
    """
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    folder = os.path.dirname(csv_path)
    if 'abstract' in column.lower():
        return os.path.join(folder, f"{base_name}_embeddings_abs_meta.pkl")
    else:
        return os.path.join(folder, f"{base_name}_embeddings_cc_meta.pkl")

def build_or_load_faiss_index(csv_path, column='Abstracts'):
    """
    Builds or loads a FAISS index for the given CSV & column.
    
    Returns:
      - faiss_index: the loaded/created FAISS index
      - embeddings:  (N, dim) float32 array of all row embeddings
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in CSV '{csv_path}'")

    index_file = get_faiss_index_file(csv_path, column)
    meta_file = get_faiss_meta_file(csv_path, column)

    model = SentenceTransformer('sentence-transformers/gtr-t5-xl', device='cpu')
    # model = SentenceTransformer('sentence-transformers/gtr-t5-xl')

    # We will either load or create:
    #  1) a faiss index file (index_file)
    #  2) a meta pickle with {num_rows, dimension} (meta_file)

    need_rebuild = True
    if os.path.exists(index_file) and os.path.exists(meta_file):
        # Check if existing index matches row count
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
        saved_num_rows = meta['num_rows']
        saved_dim = meta['dim']

        if saved_num_rows == len(df):
            # Potentially no rebuild needed
            index = faiss.read_index(index_file)
            if index.ntotal == len(df) and index.d == saved_dim:
                need_rebuild = False
                # Reconstruct embeddings for possible further use
                embeddings = []
                for i in range(len(df)):
                    vec = np.zeros(saved_dim, dtype=np.float32)
                    index.reconstruct(i, vec)
                    embeddings.append(vec)
                embeddings = np.array(embeddings, dtype=np.float32)
                return index, embeddings

    # If we need to rebuild:
    texts = df[column].fillna('').tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    num_rows = embeddings.shape[0]

    # Normalize each embedding (for cos-sim in an IP index)
    for i in range(num_rows):
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] = embeddings[i] / norm

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_file)
    meta = {'num_rows': num_rows, 'dim': dim}
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)

    return index, embeddings
