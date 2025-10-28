########################################################
# imports
########################################################

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import os
import re
from werkzeug.utils import secure_filename
from sklearn.decomposition import PCA
from embeddings import compute_embeddings
from chatbot import get_chatbot_response
from preproc import update_csv
from auth import setup_hf_auth

########################################################
# auth/bootstrap
########################################################

# ensure HF auth is set up before any model is instantiated
setup_hf_auth()


########################################################
# flask app setup
########################################################

app = Flask(__name__)
app.secret_key = "b8f2e9c1a7d54f04b0a1e37c9b6f8d2e"


########################################################
# variables
########################################################

UPLOAD_FOLDER = 'uploads'
UPDATES_FOLDER = 'updates'
DATA_FOLDER = 'data'
default_csv = f'{DATA_FOLDER}/mycobacterium_proteome_df.csv'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPDATES_FOLDER, exist_ok=True)

# Allowed extensions for CSV files
ALLOWED_EXTENSIONS = {'csv'}

# UI state
selected_csv_filename = 'data/mycobacterium_proteome_df.csv'
plot_type = '2D UMAP Based'        # default plot type
info_source = 'Function [CC]'      # default info source

# single, uniform marker size for all nodes
UNIFORM_NODE_SIZE = 10


########################################################
# helper functions
########################################################

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_available_csv_files():
    csv_files = []
    default_path = default_csv
    if os.path.exists(default_path):
        csv_files.append(('mycobacterium_proteome_df.csv', default_path))

    if os.path.isdir(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            if f.lower().endswith('.csv'):
                full_path = os.path.join(UPLOAD_FOLDER, f)
                csv_files.append((f"[Uploaded] {f}", full_path))

    if os.path.isdir(UPDATES_FOLDER):
        for f in os.listdir(UPDATES_FOLDER):
            if f.lower().endswith('.csv'):
                full_path = os.path.join(UPDATES_FOLDER, f)
                csv_files.append((f"[Updated] {f}", full_path))

    return csv_files


@app.route('/', methods=['GET', 'POST'])
def index():
    global selected_csv_filename, plot_type, info_source

    if request.method == 'POST':
        # handle CSV upload
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                upload_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(upload_path)

                # ensure required columns exist (create empty if missing)
                required_cols = ["InterPro", "PubMed ID", "Function [CC]", "Abstracts"]
                try:
                    df_test = pd.read_csv(upload_path)
                except Exception as e:
                    flash(f"Error reading CSV: {e}", "error")
                    return redirect(url_for('index'))

                missing = [c for c in required_cols if c not in df_test.columns]
                if missing:
                    flash(f"The uploaded CSV is missing required columns: {missing}", "error")
                    for mc in missing:
                        df_test[mc] = ""
                    df_test.to_csv(upload_path, index=False)
                    flash("Missing columns added as empty. CSV updated.", "info")

                flash(f"File '{filename}' uploaded successfully!", "success")
                selected_csv_filename = upload_path
            else:
                flash("Invalid file format. Only CSV files are allowed.", "error")
            return redirect(url_for('index'))

        # handle CSV selection from dropdown
        if 'selected_csv' in request.form:
            selected_csv_path = request.form.get('selected_csv')
            if selected_csv_path and os.path.exists(selected_csv_path):
                selected_csv_filename = selected_csv_path

        # plot controls (only plot type + info source now)
        plot_type = request.form.get('plot_type', plot_type)
        info_source = request.form.get('info_source', info_source)

    # read chosen CSV (fallback to default if missing)
    if not os.path.exists(selected_csv_filename):
        flash(f"Selected file {selected_csv_filename} not found. Reverting to default CSV.", "error")
        selected_csv_filename = os.path.join(DATA_FOLDER, 'mycobacterium_proteome_df.csv')

    try:
        df = pd.read_csv(selected_csv_filename)
    except Exception as e:
        flash(f"Error reading selected CSV ({selected_csv_filename}): {e}", "error")
        selected_csv_filename = os.path.join(DATA_FOLDER, 'mycobacterium_proteome_df.csv')
        df = pd.read_csv(selected_csv_filename)

    # warn if chosen info_source missing
    if info_source not in df.columns:
        flash(f"The column '{info_source}' does not exist in the current CSV.", "error")

    # clean Organism text & set categorical colors
    def clean_text(text):
        text = re.sub(r'\d+', '', str(text))
        while '(' in text and ')' in text:
            text = re.sub(r'\([^()]*\)', '', text)
        return text.strip()

    if 'Organism' in df.columns:
        df['Organism'] = df['Organism'].apply(clean_text)
        organisms = df['Organism'].fillna('Unknown').unique()
        # categorical palette (multi-color)
        color_palette = [
            '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
            '#17becf', '#1f77b4'
        ]
        color_mapping = {org: color_palette[i % len(color_palette)] for i, org in enumerate(organisms)}
        df['Color'] = df['Organism'].map(color_mapping)
    else:
        # fallback palette by cluster, if available
        labels = df['Cluster Label'].fillna('Unknown').astype(str).unique() if 'Cluster Label' in df.columns else ['All']
        color_palette = [
            '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
            '#17becf', '#1f77b4'
        ]
        color_mapping = {lab: color_palette[i % len(color_palette)] for i, lab in enumerate(labels)}
        df['Color'] = df.get('Cluster Label', pd.Series(['All'] * len(df))).astype(str).map(color_mapping)

    # embeddings (for 3D coordinates if requested)
    embeddings = None
    if info_source in df.columns:
        embeddings = compute_embeddings(compute_embeddings_csv=selected_csv_filename,column=info_source)
    else:
        flash(f"Cannot compute embeddings for '{info_source}' since it's missing in the CSV.", "warning")

    coords = None
    if embeddings is not None and plot_type.startswith('3D'):
        coords_file = os.path.join(DATA_FOLDER, f"coordinates_{info_source}.npy")
        if os.path.exists(coords_file):
            coords = np.load(coords_file)
        else:
            pca = PCA(n_components=3)
            coords = pca.fit_transform(embeddings)
            np.save(coords_file, coords)

    # build nodes (uniform size, multicolor only — no blue overlay layer anywhere)
    nodes = []
    for i in range(len(df)):
        row = df.iloc[i]
        protein = row.get('Protein names', 'N/A')
        organism = row.get('Organism', 'N/A')
        gene = row.get('Gene Names', 'N/A')
        pathway = row.get('Pathway', 'N/A')
        anot = row.get('Annotation', 'N/A')
        label = row.get('Cluster Label', 'N/A')
        color = row.get('Color', '#7f7f7f')  # categorical color

        hover = (
            f"Protein Names: {protein}<br>Organism: {organism}<br>Gene Names: {gene}<br>"
            f"Pathway: {pathway}<br>Annotation: {anot}<br>Cluster: {label}"
        )

        if plot_type.startswith('3D') and coords is not None and i < len(coords):
            x, y, z = coords[i]
            node = {
                'id': int(i),
                'protein_name': protein,
                'label': hover,
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'group': str(label),
                'size': float(UNIFORM_NODE_SIZE),
                'color': color,
            }
        else:
            x2d = row.get('UMAP 1', 0.0)
            y2d = row.get('UMAP 2', 0.0)
            node = {
                'id': int(i),
                'protein_name': protein,
                'label': hover,
                'x': float(x2d),
                'y': float(y2d),
                'group': str(label),
                'size': float(UNIFORM_NODE_SIZE),
                'color': color,
            }

        nodes.append(node)

    edges = []  # still no edges in this view

    # plot & info options
    plot_options = ['2D UMAP Based', '3D PCA Based']
    info_options = ['Abstracts', 'Function [CC]']

    # list of CSVs for dropdown
    csv_file_choices = get_available_csv_files()

    return render_template(
        'index.html',
        nodes=nodes,
        edges=edges,
        plot_type=plot_type,
        plot_options=plot_options,
        info_source=info_source,
        info_options=info_options,
        csv_file_choices=csv_file_choices,
        selected_csv=selected_csv_filename
    )


@app.route('/chatbot', methods=['POST'])
def chatbot():
    global info_source, selected_csv_filename
    data = request.get_json()
    node_ids = data.get('node_ids', [])
    message = data.get('message', '')
    include_similar = data.get('include_similar', True)
    response_text = get_chatbot_response(
        node_ids=node_ids,
        message=message,
        include_similar=include_similar,
        info_source=info_source,
        csv_path=selected_csv_filename
    )
    return jsonify({'message': response_text})


@app.route('/fetch_function_cc', methods=['POST'])
def fetch_function_cc():
    """
    Fetch/Update 'Function [CC]' via UniProt for missing rows in the current CSV,
    then save the updated CSV into the 'updates' folder.
    """
    global selected_csv_filename

    if not os.path.exists(selected_csv_filename):
        flash(f"Selected file {selected_csv_filename} does not exist.", "error")
        return redirect(url_for('index'))

    base_name = os.path.basename(selected_csv_filename)
    name_no_ext = os.path.splitext(base_name)[0]
    new_filename = f"{name_no_ext}_functioncc_updated.csv"
    new_filepath = os.path.join(UPDATES_FOLDER, new_filename)

    try:
        update_csv(selected_csv_filename)  # in-place update by preproc.py
        df_updated = pd.read_csv(selected_csv_filename)
        df_updated.to_csv(new_filepath, index=False)
        flash(f"Function [CC] updated and saved as '{new_filename}' in updates folder.", "success")
    except Exception as e:
        flash(f"Error updating Function [CC]: {e}", "error")

    return redirect(url_for('index'))


@app.route('/fetch_abstracts', methods=['POST'])
def fetch_abstracts():
    global selected_csv_filename
    if not os.path.exists(selected_csv_filename):
        flash(f"Selected file {selected_csv_filename} does not exist.", "error")
        return redirect(url_for('index'))

    df = pd.read_csv(selected_csv_filename)
    if 'Abstracts' not in df.columns:
        flash("The 'Abstracts' column is missing in this CSV. Cannot fetch or update abstracts.", "error")
        return redirect(url_for('index'))

    df['Abstracts'] = df['Abstracts'].fillna('No data')

    base_name = os.path.basename(selected_csv_filename)
    name_no_ext = os.path.splitext(base_name)[0]
    new_filename = f"{name_no_ext}_abstracts_updated.csv"
    new_filepath = os.path.join(UPDATES_FOLDER, new_filename)
    df.to_csv(new_filepath, index=False)

    flash(f"Abstracts updated (placeholder) and saved as '{new_filename}' in updates folder.", "success")
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, port=0)

# if __name__ == "__main__":
#     port = int(os.getenv("PORT", 5000))
#     app.run(
#         host=os.getenv("HOST", "127.0.0.1"),
#         port=port,
#         debug=os.getenv("FLASK_DEBUG", "0") == "1",
#         use_reloader=os.getenv("FLASK_USE_RELOADER", "1") == "1",
#         threaded=True,
#     )
