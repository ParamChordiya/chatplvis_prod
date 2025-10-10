import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity  # kept for compatibility if you prefer CPU path
from sentence_transformers import SentenceTransformer
import openai
import markdown

def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"

def _cosine_sim_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: (1, d) or (n, d), b: (m, d) -> returns (n, m) cosine sims
    """
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return a_n @ b_n.T

def get_chatbot_response(node_ids, message, include_similar=True,
                         info_source='Abstracts',
                         csv_path='data/mycobacterium_proteome_df.csv'):
    """
    Uses GPU (CUDA/MPS) when available for similarity computation and model loading.
    """
    if not os.path.exists(csv_path):
        return "Error: The data file does not exist. Please select a valid CSV."

    df = pd.read_csv(csv_path)

    if info_source not in df.columns:
        return f"Error: The column '{info_source}' is not present in the selected CSV."

    # Embeddings file by column
    if info_source == 'Abstracts':
        embeddings_file = os.path.join('data', 'embeddings_abs.npy')
        text_column = 'Abstracts'
    else:
        embeddings_file = os.path.join('data', 'embeddings_cc.npy')
        text_column = 'Function [CC]'

    if not os.path.exists(embeddings_file):
        return f"Error: The embeddings file '{embeddings_file}' does not exist. Please compute embeddings first."

    embeddings = np.load(embeddings_file)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    # Choose device
    device = _best_device()

    # Load ST model on chosen device (kept in case you later encode the user query, etc.)
    try:
        st_model = SentenceTransformer('sentence-transformers/gtr-t5-xl', device=device)
    except Exception as e:
        # Don’t fail hard; continue without using the model
        st_model = None
        print(f"[warn] Failed to load SentenceTransformer on {device}: {e}")

    # Move embeddings to torch on the chosen device for fast cosine sim
    emb_t = torch.from_numpy(embeddings.astype(np.float32, copy=False)).to(device)  # (N, D)

    context = ""
    for node_id in node_ids:
        if node_id < 0 or node_id >= len(df):
            continue

        clicked_protein_name = df.iloc[node_id].get('Protein names', 'Unknown')
        clicked_protein_info = df.iloc[node_id].get(text_column, '')
        context += f"Selected Protein Name: {clicked_protein_name}\n"
        context += f"{info_source}: {clicked_protein_info}\n\n"

        if include_similar and node_id < emb_t.shape[0]:
            # (1, D)
            clicked_vec = emb_t[node_id:node_id+1, :]

            # GPU cosine similarity
            try:
                sims_t = _cosine_sim_torch(clicked_vec, emb_t).squeeze(0)  # (N,)
                sims = sims_t.detach().cpu().numpy()
            except Exception as e:
                # Fallback to CPU sklearn if something odd happens
                print(f"[warn] Torch cosine sim failed on {device}: {e}, falling back to CPU.")
                sims = cosine_similarity(
                    embeddings[node_id].reshape(1, -1), embeddings
                )[0]

            K = 6
            top_k_indices = sims.argsort()[-K:][::-1]
            top_k_indices = [idx for idx in top_k_indices if idx != node_id][:5]

            for idx in top_k_indices:
                if idx < len(df):
                    protein_name = df.iloc[idx].get('Protein names', 'Unknown')
                    related_info = df.iloc[idx].get(text_column, '')
                    context += f"Related Protein Name: {protein_name}\n{info_source}: {related_info}\n\n"

    prompt = f"{context}\nUser Question: {message}\nAssistant:"

    openai.api_key = os.getenv('OPENAI_API_KEY')
    if openai.api_key is None:
        return "Error: OpenAI API key not set in environment."

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an expert assistant providing detailed information about proteins. "
                "You are being provided a list of proteins along with functional annotations (Context) from UniProt. "
                "Use that context plus any background knowledge to give an overview of these proteins."
            )},
            {"role": "user", "content": prompt}
        ]
    )

    assistant_reply = response['choices'][0]['message']['content']
    assistant_html = markdown.markdown(assistant_reply, extensions=['extra', 'nl2br'])
    return assistant_html
