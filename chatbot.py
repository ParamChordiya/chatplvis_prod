import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
import markdown

from dotenv import load_dotenv

# Load .env file
load_dotenv()

from embeddings import build_or_load_faiss_index, get_faiss_index_file, get_faiss_meta_file

def get_chatbot_response(node_ids, message, include_similar=True,
                         info_source='Abstracts', csv_path='data/mtuberculosis_df_abs.csv'):
    """
    Main function to return chatbot response using a FAISS-based retrieval approach,
    now using 'abs_id' to look up the row(s).

    Steps:
      1) Load the CSV & build/load the FAISS index for that CSV & column.
      2) Embed the user's question with SentenceTransformers.
      3) FAISS search to find top-K relevant rows (if include_similar=True).
      4) Also forcibly include the user-selected node(s) in the context (via 'abs_id').
      5) Construct a prompt with the chosen context + the user’s question.
      6) Call openai with model="gpt-4o" and return the response in HTML.
    """
    if not os.path.exists(csv_path):
        return "Error: The data file does not exist. Please select a valid CSV."

    df = pd.read_csv(csv_path)
    if info_source not in df.columns:
        return f"Error: The column '{info_source}' is not present in the selected CSV."

    # We'll create an index on abs_id for easy row lookups
    if 'abs_id' not in df.columns:
        return "Error: The CSV does not contain 'abs_id' column. Please regenerate or fix the CSV."

    df_indexed = df.set_index('abs_id', drop=False)

    # Build or load the Faiss index
    try:
        faiss_index, embeddings = build_or_load_faiss_index(csv_path, info_source)
    except Exception as e:
        return f"Error building/loading FAISS index: {str(e)}"

    # If the index is empty, skip
    if faiss_index.ntotal == 0:
        return "Error: No rows in the dataset or the FAISS index is empty."

    # We'll embed the question with the same model
    # model = SentenceTransformer('sentence-transformers/gtr-t5-xl')
    model = SentenceTransformer("sentence-transformers/gtr-t5-xl", device="cpu")

    q_emb = model.encode([message])[0].astype(np.float32)

    # Normalize question embedding for cos-sim if needed
    norm_q = np.linalg.norm(q_emb)
    if norm_q > 0:
        q_emb = q_emb / norm_q

    # Search top-K (if include_similar=True)
    top_k = 6
    q_emb_2d = q_emb.reshape(1, -1)
    distances, indices = faiss_index.search(q_emb_2d, top_k)
    best_idxs = indices[0].tolist()

    # Build context
    context = ""
    added_indices = set()

    # 1) Always add user-selected node(s) explicitly
    for node_id in node_ids:
        if node_id in df_indexed.index:
            if node_id not in added_indices:
                added_indices.add(node_id)
                clicked_row = df_indexed.loc[node_id]
                clicked_protein_name = clicked_row.get('Protein names', 'Unknown')
                clicked_protein_info = clicked_row.get(info_source, '')
                context += f"Selected Protein Name: {clicked_protein_name}\n"
                context += f"{info_source}: {clicked_protein_info}\n\n"

    # 2) If user wants top-k similar, add them
    if include_similar:
        # The FAISS search 'best_idxs' are row positions, not 'abs_id' values.
        # So we must locate each row by index position in the original DataFrame.
        # Because we called 'build_or_load_faiss_index' which uses df order,
        # index i corresponds to df.iloc[i].
        for idx in best_idxs:
            if idx < 0 or idx >= len(df):
                continue
            # Get the abs_id from that row
            candidate_abs_id = df.iloc[idx]['abs_id']
            if candidate_abs_id not in added_indices:
                added_indices.add(candidate_abs_id)
                protein_name = df.iloc[idx].get('Protein names', 'Unknown')
                row_info = df.iloc[idx].get(info_source, '')
                context += f"Related Protein Name: {protein_name}\n"
                context += f"{info_source}: {row_info}\n\n"

    # Final prompt
    prompt = f"{context}\nUser Question: {message}\nAssistant:"

    # Set up OpenAI
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        return "Error: OpenAI API key not set in environment."

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Think step by step before responding."
                    "You are an expert assistant providing detailed information about proteins. "
                    "You are given a set of protein data (Abstracts, Function [CC], etc.). "
                    "Use that context plus your own knowledge to answer the user's question comprehensively."
                    # "PLEASE GIVE YOUR OUTPUT AS A WELL STRUCTURED MARKDOWN CODE ONLY and end with '```'"
                )
            },
            {"role": "user", "content": prompt}
        ]
    )

    assistant_reply = response['choices'][0]['message']['content']
    # Convert Markdown to HTML
    assistant_html = markdown.markdown(assistant_reply, extensions=['extra', 'nl2br'])
    # assistant_html = markdown.markdown(
        # assistant_reply.strip("```"),  # Strip trailing and leading code blocks
        # extensions=['extra', 'nl2br', 'fenced_code', 'tables', 'toc', 'abbr']
    # )
    return assistant_html


def baseline_get_response(message):
    """
    A baseline with no additional context, just direct question -> GPT.
    Using 'gpt-4o' for consistency.
    """
    import openai, markdown, os
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        return "Error: OpenAI API key not set."

    prompt = f"User Question: {message}\nAssistant:"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    reply = response['choices'][0]['message']['content']
    return markdown.markdown(reply, extensions=['extra', 'nl2br'])
