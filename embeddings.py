import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def compute_embeddings(column='Abstracts'):
    # Depending on the chosen column, load or compute embeddings
    # We'll store separate embeddings for Abstracts and Function [CC]
    if column == 'Abstracts':
        embeddings_file = 'data/embeddings_abs.npy'
    else:
        embeddings_file = 'data/embeddings_cc.npy'

    model = SentenceTransformer('sentence-transformers/gtr-t5-xl', device='cpu')

    if os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file)
    else:
        df = pd.read_csv('data/mtuberculosis_df_abs.csv')
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in dataframe.")
        texts = df[column].fillna('').tolist()
        embeddings = model.encode(texts, show_progress_bar=True)
        np.save(embeddings_file, embeddings)

    return embeddings
