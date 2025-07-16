
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
import markdown


csv_path = 'data/mtuberculosis_df_abs.csv'

def get_chatbot_response(node_ids, message, include_similar=True, info_source='Abstracts', csv_path=csv_path):
    """
    csv_path: the path to the currently active CSV file
    """
    if not os.path.exists(csv_path):
        return "Error: The data file does not exist. Please select a valid CSV."

    df = pd.read_csv(csv_path)

    # If the chosen info_source is missing in the CSV, return an error
    if info_source not in df.columns:
        return f"Error: The column '{info_source}' is not present in the selected CSV."

    # Decide which embeddings file to load
    if info_source == 'Abstracts':
        embeddings_file = os.path.join('data', 'embeddings_abs.npy')
        text_column = 'Abstracts'
    else:
        embeddings_file = os.path.join('data', 'embeddings_cc.npy')
        text_column = 'Function [CC]'

    if not os.path.exists(embeddings_file):
        return f"Error: The embeddings file '{embeddings_file}' does not exist. Please compute embeddings first."

    embeddings = np.load(embeddings_file)

    # Ensure embeddings shape
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)

    model = SentenceTransformer('sentence-transformers/gtr-t5-xl', device='cpu')

    context = ""
    for node_id in node_ids:
        # If the node index is out of range, skip
        if node_id < 0 or node_id >= len(df):
            continue

        clicked_protein_name = df.iloc[node_id].get('Protein names', 'Unknown')
        clicked_protein_info = df.iloc[node_id].get(text_column, '')
        context += f"Selected Protein Name: {clicked_protein_name}\n"
        context += f"{info_source}: {clicked_protein_info}\n\n"

        if include_similar and node_id < len(embeddings):
            clicked_protein_embedding = embeddings[node_id].reshape(1, -1)
            similarities = cosine_similarity(clicked_protein_embedding, embeddings)[0]

            K = 6
            top_k_indices = similarities.argsort()[-K:][::-1]
            top_k_indices = [idx for idx in top_k_indices if idx != node_id][:5]

            for idx in top_k_indices:
                if idx < len(df):
                    protein_name = df.iloc[idx].get('Protein names', 'Unknown')
                    related_info = df.iloc[idx].get(text_column, '')
                    context += f"Related Protein Name: {protein_name}\n{info_source}: {related_info}\n\n"

    prompt = f"{context}\nUser Question: {message}\nAssistant:"

    openai.api_key = os.getenv('OPENAI_API_KEY')
    # openai.api_key =  apikey
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
    # Convert Markdown to HTML
    assistant_html = markdown.markdown(assistant_reply, extensions=['extra', 'nl2br'])
    return assistant_html