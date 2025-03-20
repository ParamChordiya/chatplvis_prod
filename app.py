# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import re

from multiprocessing import Pool

pool = Pool()
# Do work with pool
pool.close()
pool.join()

# import psutil
# import GPUtil

from chatbot import get_chatbot_response
from embeddings import build_or_load_faiss_index
from sklearn.decomposition import PCA

app = Flask(__name__)

# ----------------------------------------------------------------
# SINGLE CSV PATH (the only CSV used in the app)
# ----------------------------------------------------------------
CSV_PATH = 'uploads/mycobacterium_proteome_df.csv'

# Default user selections
sel_col = 'Total_Counts'
sel_comp = 'All proteomes'
plot_type = '2D'                  # default plot type
info_source = 'Function [CC]'     # default info source

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Render the main page and handle plot-related form submissions.
    We read only from CSV_PATH = 'uploads/mycobacterium_proteome_df.csv'.
    """
    global sel_col, sel_comp, plot_type, info_source

    # Handle user form submission for the plot settings
    if request.method == 'POST':
        sel_col = request.form.get('sel_col', sel_col)
        sel_comp = request.form.get('sel_comp', sel_comp)
        plot_type = request.form.get('plot_type', plot_type)
        info_source = request.form.get('info_source', info_source)

    # Read the single CSV
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Error: The CSV file '{CSV_PATH}' does not exist. "
            "Please place it in the uploads folder."
        )

    try:
        mycobacterium_df = pd.read_csv(CSV_PATH)
    except Exception as e:
        raise RuntimeError(f"Error reading CSV '{CSV_PATH}': {e}")

    # Ensure we have a stable abs_id column
    if 'abs_id' not in mycobacterium_df.columns:
        mycobacterium_df['abs_id'] = range(len(mycobacterium_df))
        mycobacterium_df.to_csv(CSV_PATH, index=False)

    def clean_text(text):
        text = re.sub(r'\d+', '', str(text))
        while '(' in text and ')' in text:
            text = re.sub(r'\([^()]*\)', '', text)
        return text.strip()

    # Clean "Organism" if it exists
    if 'Organism' in mycobacterium_df.columns:
        mycobacterium_df['Organism'] = mycobacterium_df['Organism'].apply(clean_text)
        organisms = mycobacterium_df['Organism'].unique()

        color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf'
        ]
        color_mapping = {
            organism: color_palette[i % len(color_palette)]
            for i, organism in enumerate(organisms)
        }
        mycobacterium_df['Color'] = mycobacterium_df['Organism'].map(color_mapping)
    else:
        # Default color
        mycobacterium_df['Color'] = '#1f77b4'

    # Optionally load counts from Excel
    counts_path = os.path.join('data', 'counts_all_stages_MAGECK_with_ES.xlsx')
    if os.path.exists(counts_path):
        counts_df = pd.read_excel(counts_path)
        counts_df['clean_orf'] = counts_df['orf'].apply(
            lambda x: re.sub(r'(?<=RV)BD', '', str(x), count=1)
        )
        counts_df['clean_name'] = counts_df['name'].apply(
            lambda x: re.sub(r'(?<=RV)BD', '', str(x), count=1)
        )
    else:
        counts_df = pd.DataFrame()

    # Separate out the M. tuberculosis rows (if “Organism” is present)
    if 'Organism' in mycobacterium_df.columns:
        tb_df = mycobacterium_df[mycobacterium_df['Organism'] == 'Mycobacterium tuberculosis'].copy()
    else:
        tb_df = mycobacterium_df.copy()

    # Ensure counts columns exist
    cols = ['Counts_1st', 'Counts_2nd', 'Counts_3rd', 'Total_Counts', 'Rank']
    import numpy as np
    for c in cols:
        if c not in tb_df.columns:
            tb_df[c] = np.nan

    # Merge counts if available
    if not counts_df.empty:
        for _, row in counts_df.iterrows():
            clean_orf_escaped = re.escape(str(row['clean_orf']))
            clean_name_escaped = re.escape(str(row['clean_name']))
            pattern = f"{clean_orf_escaped}|{clean_name_escaped}"
            if 'Gene Names' in tb_df.columns:
                mask = tb_df['Gene Names'].str.contains(pattern, case=False, na=False, regex=True)
                tb_df.loc[mask, cols] = [
                    row['Counts_1st'],
                    row['Counts_2nd'],
                    row['Counts_3rd'],
                    row['Total_Counts'],
                    row['Rank']
                ]

    # Normalize them if needed
    def normalize_columns(df, columns):
        for column_name in columns:
            min_val = df[column_name].min()
            max_val = df[column_name].max()
            if max_val - min_val == 0:
                df[column_name + '_normalized'] = 0
            else:
                df[column_name + '_normalized'] = (
                    (df[column_name] - min_val) / (max_val - min_val)
                )
        return df

    tb_df = normalize_columns(tb_df, cols)

    # Ensure sel_col is valid
    if sel_col not in tb_df.columns:
        # fallback to a normalized column if possible
        possible_norm = [c for c in tb_df.columns if c.endswith('_normalized')]
        if possible_norm:
            sel_col = possible_norm[0]
        else:
            sel_col = tb_df.columns[0]

    tb_df_subset = tb_df[['Entry', sel_col, 'Rank', 'abs_id']].copy()

    # Build plot_df based on user selection
    if sel_comp == 'All proteomes':
        mycobacterium_df_subset = mycobacterium_df.copy()
        plot_df = pd.merge(
            mycobacterium_df_subset,
            tb_df_subset,
            how='left',
            on=['Entry','abs_id'],
            suffixes=('', '_tb')
        )
        plot_df[sel_col] = plot_df[sel_col].fillna(0)
    elif sel_comp == 'Mycobacterium tuberculosis':
        plot_df = tb_df.copy()
        plot_df[sel_col] = plot_df[sel_col].fillna(0)
    else:
        # e.g., "vs smegmatis"
        other_org = sel_comp[3:]
        if 'Organism' in mycobacterium_df.columns:
            mycobacterium_df_subset = mycobacterium_df[
                (mycobacterium_df['Organism'] == 'Mycobacterium tuberculosis')
                | (mycobacterium_df['Organism'].str.contains(other_org, case=False, na=False))
            ]
        else:
            mycobacterium_df_subset = mycobacterium_df.copy()

        plot_df = pd.merge(
            mycobacterium_df_subset,
            tb_df_subset,
            how='left',
            on=['Entry','abs_id'],
            suffixes=('', '_tb')
        )
        plot_df[sel_col] = plot_df[sel_col].fillna(0)

    # ----------------------------------------------------------------
    # SET ALL NODE SIZES TO A SINGLE CONSTANT (no expression-based scaling)
    # ----------------------------------------------------------------
    plot_df['Size'] = 10  # pick any fixed size you like, e.g. 10

    # Build or load the FAISS index for the chosen info_source
    embeddings = None
    faiss_index = None
    if info_source in mycobacterium_df.columns:
        try:
            faiss_index, embeddings = build_or_load_faiss_index(CSV_PATH, info_source)
        except Exception:
            pass

    # If 3D plot, do PCA on the embeddings if present
    coords = None
    if embeddings is not None and plot_type.startswith('3D'):
        pca = PCA(n_components=3)
        coords = pca.fit_transform(embeddings)

    # Build node objects
    nodes = []
    for i in range(len(plot_df)):
        row = plot_df.iloc[i]
        protein = row.get('Protein names', 'N/A')
        organism = row.get('Organism', 'N/A')
        gene = row.get('Gene Names', 'N/A')
        pathway = row.get('Pathway', 'N/A')
        anot = row.get('Annotation', 'N/A')
        counts_val = row.get(sel_col, 'N/A')
        size = row.get('Size', 10)
        label = row.get('Cluster Label', 'N/A')
        color = row.get('Color', '#1f77b4')
        abs_id = row.get('abs_id', i)

        # Build hover text
        text = (
            f"Protein Names: {protein}<br>"
            f"Organism: {organism}<br>"
            f"Gene Names: {gene}<br>"
            f"Pathway: {pathway}<br>"
            f"Counts: {counts_val}<br>"
            f"Annotation: {anot}<br>"
            f"Cluster: {label}"
        )

        if coords is not None and i < len(coords) and plot_type.startswith('3D'):
            x, y, z = coords[i]
            node = {
                'id': int(abs_id),
                'protein_name': protein,
                'label': text,
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'group': str(label),
                'size': float(size),
                'color': color,
            }
        else:
            x2d = row.get('UMAP 1', 0.0)
            y2d = row.get('UMAP 2', 0.0)
            node = {
                'id': int(abs_id),
                'protein_name': protein,
                'label': text,
                'x': float(x2d),
                'y': float(y2d),
                'group': str(label),
                'size': float(size),
                'color': color,
            }
        nodes.append(node)

    edges = []  # no edges in this example

    # Build dynamic dropdown options
    column_options = [
        'Counts_1st_normalized',
        'Counts_2nd_normalized',
        'Counts_3rd_normalized',
        'Total_Counts_normalized',
        'Rank_normalized'
    ]
    for c in cols:
        if c in tb_df.columns and c not in column_options:
            column_options.append(c)

    comparison_options = [
        'All proteomes',
        'Mycobacterium tuberculosis',
        'vs smegmatis',
        'vs marinum',
        'vs leprae',
        'vs kansasii',
        'vs intracellulare',
        'vs fortuitum',
        'vs bovis'
    ]
    plot_options = ['2D UMAP Based', '3D PCA Based']
    info_options = ['Function [CC]', 'Abstracts']

    return render_template(
        'index.html',
        nodes=nodes,
        edges=edges,
        sel_col=sel_col,
        sel_comp=sel_comp,
        column_options=column_options,
        comparison_options=comparison_options,
        plot_type=plot_type,
        plot_options=plot_options,
        info_source=info_source,
        info_options=info_options
    )


@app.route('/chatbot', methods=['POST'])
def chatbot():
    """
    Chatbot endpoint. If no node is selected (node_ids is empty), respond with
    'No node was selected.' Otherwise, proceed with FAISS-based lookup.
    """
    data = request.get_json()
    node_ids = data.get('node_ids', [])
    message = data.get('message', '')
    include_similar = data.get('include_similar', True)

    if not node_ids:
        return jsonify({'message': "No node was selected."})

    # Convert node_ids to integer if needed
    node_ids_int = []
    for n in node_ids:
        try:
            node_ids_int.append(int(n))
        except ValueError:
            continue

    response_text = get_chatbot_response(
        node_ids=node_ids_int,
        message=message,
        include_similar=include_similar,
        info_source=info_source,
        csv_path=CSV_PATH
    )
    return jsonify({'message': response_text})


# @app.route("/system_info")
# def system_info():
#     # CPU cores
#     total_cores = psutil.cpu_count(logical=True)   # total logical cores
#     physical_cores = psutil.cpu_count(logical=False)
    
#     # CPU usage (as a percentage)
#     cpu_usage_percent = psutil.cpu_percent(interval=0.5)
    
#     # Memory usage
#     memory_info = psutil.virtual_memory()
#     total_memory_gb = round(memory_info.total / (1024**3), 2)
#     used_memory_gb = round(memory_info.used / (1024**3), 2)
#     memory_usage_percent = memory_info.percent

#     # GPU info (for NVIDIA GPUs)
#     gpus = GPUtil.getGPUs()
#     gpu_list = []
#     for gpu in gpus:
#         gpu_info = {
#             "id": gpu.id,
#             "name": gpu.name,
#             "load_percent": round(gpu.load * 100, 2),
#             "memory_total_gb": round(gpu.memoryTotal / 1024, 2),
#             "memory_used_gb": round(gpu.memoryUsed / 1024, 2),
#             "memory_free_gb": round(gpu.memoryFree / 1024, 2),
#             "temperature_c": gpu.temperature
#         }
#         gpu_list.append(gpu_info)

#     # Build a JSON response
#     response_data = {
#         "cpu": {
#             "logical_cores": total_cores,
#             "physical_cores": physical_cores,
#             "usage_percent": cpu_usage_percent
#         },
#         "memory": {
#             "total_gb": total_memory_gb,
#             "used_gb": used_memory_gb,
#             "usage_percent": memory_usage_percent
#         },
#         "gpus": gpu_list
#     }
    
#     return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port='0')
