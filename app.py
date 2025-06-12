from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import os
import re
from werkzeug.utils import secure_filename

from embeddings import compute_embeddings
from chatbot import get_chatbot_response
from sklearn.decomposition import PCA

from preproc import update_csv

app = Flask(__name__)
app.secret_key = "some_secret_key_for_sessions"  # needed for flashing messages

# Folders configuration
UPLOAD_FOLDER = 'uploads'
UPDATES_FOLDER = 'updates'
DATA_FOLDER = 'data'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPDATES_FOLDER, exist_ok=True)

# Allowed extensions for CSV files
ALLOWED_EXTENSIONS = {'csv'}

# Global variables to store user selections
#selected_csv_filename = 'mtuberculosis_df_abs.csv'  # default CSV in 'data' folder
selected_csv_filename = 'mycobacterium_proteome_df.csv'
sel_col = 'Total_Counts'
sel_comp = 'All proteomes'
plot_type = '2D'           # default plot type
info_source = 'Abstracts'  # default info source


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_available_csv_files():
    """
    Returns a list of all CSV files that the user can choose from:
      - The default CSV from DATA_FOLDER
      - Any files in UPLOAD_FOLDER
      - Any updated CSVs in UPDATES_FOLDER
    The returned list is a list of tuples: (label_for_dropdown, full_path).
    """
    csv_files = []

    # 1) Default in data folder
    default_path = os.path.join(DATA_FOLDER, 'mycobacterium_proteome_df.csv')
    if os.path.exists(default_path):
        csv_files.append(('Default: mycobacterium_proteome_df.csv', default_path))

    # 2) All uploaded files
    for f in os.listdir(UPLOAD_FOLDER):
        if f.lower().endswith('.csv'):
            full_path = os.path.join(UPLOAD_FOLDER, f)
            csv_files.append((f"[Uploaded] {f}", full_path))

    # 3) All updated files
    for f in os.listdir(UPDATES_FOLDER):
        if f.lower().endswith('.csv'):
            full_path = os.path.join(UPDATES_FOLDER, f)
            csv_files.append((f"[Updated] {f}", full_path))

    return csv_files


@app.route('/', methods=['GET', 'POST'])
def index():
    global selected_csv_filename, sel_col, sel_comp, plot_type, info_source

    # Handle form submissions (CSV upload, CSV selection, plot updates)
    if request.method == 'POST':
        # 1) Check if we are uploading a file
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                upload_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(upload_path)

                # Now check for required columns
                required_cols = ["InterPro", "PubMed ID", "Function [CC]", "Abstracts"]
                try:
                    df_test = pd.read_csv(upload_path)
                except Exception as e:
                    flash(f"Error reading CSV: {e}", "error")
                    return redirect(url_for('index'))

                missing = [c for c in required_cols if c not in df_test.columns]
                if missing:
                    flash(f"The uploaded CSV is missing required columns: {missing}", "error")
                    # Optionally add them as empty columns
                    for mc in missing:
                        df_test[mc] = ""
                    df_test.to_csv(upload_path, index=False)
                    flash(f"Missing columns added as empty. CSV updated.", "info")

                flash(f"File '{filename}' uploaded successfully!", "success")
                # Set the newly uploaded file as the selected CSV
                selected_csv_filename = upload_path
            else:
                flash("Invalid file format. Only CSV files are allowed.", "error")

            return redirect(url_for('index'))

        # 2) If not uploading a file, maybe user changed the selected CSV from dropdown
        if 'selected_csv' in request.form:
            selected_csv_path = request.form.get('selected_csv')
            if selected_csv_path and os.path.exists(selected_csv_path):
                selected_csv_filename = selected_csv_path

        # 3) Plot updates
        sel_col = request.form.get('sel_col', sel_col)
        sel_comp = request.form.get('sel_comp', sel_comp)
        plot_type = request.form.get('plot_type', plot_type)
        info_source = request.form.get('info_source', info_source)

    # Attempt to read the selected CSV
    if not os.path.exists(selected_csv_filename):
        flash(f"Selected file {selected_csv_filename} not found. Reverting to default CSV.", "error")
        selected_csv_filename = os.path.join(DATA_FOLDER, 'mycobacterium_proteome_df.csv')

    try:
        mycobacterium_df = pd.read_csv(selected_csv_filename)
    except Exception as e:
        flash(f"Error reading selected CSV ({selected_csv_filename}): {e}", "error")
        # Fallback to default
        selected_csv_filename = os.path.join(DATA_FOLDER, 'mycobacterium_proteome_df.csv')
        mycobacterium_df = pd.read_csv(selected_csv_filename)

    # If user picks an info_source that doesn't exist in the current CSV, warn them
    if info_source not in mycobacterium_df.columns:
        flash(f"The column '{info_source}' does not exist in the current CSV.", "error")

    def clean_text(text):
        text = re.sub(r'\d+', '', str(text))
        while '(' in text and ')' in text:
            text = re.sub(r'\([^()]*\)', '', text)
        return text.strip()

    def normalize_columns(df, columns):
        for column_name in columns:
            min_val = df[column_name].min()
            max_val = df[column_name].max()
            if max_val - min_val == 0:
                df[column_name + '_normalized'] = 0
            else:
                df[column_name + '_normalized'] = (df[column_name] - min_val) / (max_val - min_val)
        return df

    # Clean "Organism" if it exists
    if 'Organism' in mycobacterium_df.columns:
        mycobacterium_df['Organism'] = mycobacterium_df['Organism'].apply(clean_text)
        organisms = mycobacterium_df['Organism'].unique()
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                         '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                         '#bcbd22', '#17becf']
        color_mapping = {organism: color_palette[i % len(color_palette)] for i, organism in enumerate(organisms)}
        mycobacterium_df['Color'] = mycobacterium_df['Organism'].map(color_mapping)
    else:
        # If there's no Organism column, just assign a default color
        mycobacterium_df['Color'] = '#1f77b4'

    # Load counts data for M. tuberculosis from Excel
    counts_path = os.path.join(DATA_FOLDER, 'counts_all_stages_MAGECK_with_ES.xlsx')
    counts_df = pd.read_excel(counts_path)
    counts_df['clean_orf'] = counts_df['orf'].apply(lambda x: re.sub(r'(?<=RV)BD', '', str(x), count=1))
    counts_df['clean_name'] = counts_df['name'].apply(lambda x: re.sub(r'(?<=RV)BD', '', str(x), count=1))

    # Filter for M. tuberculosis
    if 'Organism' in mycobacterium_df.columns:
        tb_df = mycobacterium_df[mycobacterium_df['Organism'] == 'Mycobacterium tuberculosis'].copy()
    else:
        tb_df = mycobacterium_df.copy()

    cols = ['Counts_1st', 'Counts_2nd', 'Counts_3rd', 'Total_Counts', 'Rank']
    for c in cols:
        if c not in tb_df.columns:
            tb_df[c] = np.nan

    # Fill in the numeric columns from counts_df
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

    tb_df = normalize_columns(tb_df, cols)

    # Make sure sel_col is valid; if not, pick a normalized one
    if sel_col not in tb_df.columns:
        if 'Total_Counts_normalized' in tb_df.columns:
            sel_col = 'Total_Counts_normalized'
        else:
            possible_norm = [c for c in tb_df.columns if c.endswith('_normalized')]
            sel_col = possible_norm[0] if possible_norm else tb_df.columns[0]

    tb_df_subset = tb_df[['Entry', sel_col, 'Rank']].copy()

    # Build plot_df
    if sel_comp == 'All proteomes':
        mycobacterium_df_subset = mycobacterium_df.copy()
        plot_df = pd.merge(mycobacterium_df_subset, tb_df_subset, how='left', on='Entry')
        plot_df[sel_col] = plot_df[sel_col].fillna(0)
    elif sel_comp == 'Mycobacterium tuberculosis':
        plot_df = tb_df.copy()
        plot_df[sel_col] = plot_df[sel_col].fillna(0)
    else:
        # e.g., "vs smegmatis"
        other_org = sel_comp[3:]
        if 'Organism' in mycobacterium_df.columns:
            mycobacterium_df_subset = mycobacterium_df[
                (mycobacterium_df['Organism'] == 'Mycobacterium tuberculosis') |
                (mycobacterium_df['Organism'].str.contains(other_org, case=False, na=False))
            ]
        else:
            mycobacterium_df_subset = mycobacterium_df.copy()

        plot_df = pd.merge(mycobacterium_df_subset, tb_df_subset, how='left', on='Entry')
        plot_df[sel_col] = plot_df[sel_col].fillna(0)

    # --- FIX for large node sizes: force local min-max scaling for the chosen column ---
    min_size = 10
    max_size = 50

    if sel_col in plot_df.columns:
        col_data = plot_df[sel_col].astype(float)
        col_min, col_max = col_data.min(), col_data.max()
        if col_max - col_min > 0:
            col_data_norm = (col_data - col_min) / (col_max - col_min)
        else:
            col_data_norm = 0
        # map [0,1] -> [10,50]
        plot_df['Size'] = min_size + col_data_norm * (max_size - min_size)
    else:
        # fallback if the column doesn't exist for some reason
        plot_df['Size'] = 10

    # Embeddings
    embeddings = None
    if info_source in mycobacterium_df.columns:
        embeddings = compute_embeddings(column=info_source)
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

    # Build nodes list
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

        text = (
            f"Protein Names: {protein}<br>Organism: {organism}<br>Gene Names: {gene}<br>"
            f"Pathway: {pathway}<br>Counts: {counts_val}<br>Annotation: {anot}<br>"
            f"Point Size: {size}<br>Cluster: {label}"
        )

        if plot_type.startswith('3D') and coords is not None and i < len(coords):
            x, y, z = coords[i]
            node = {
                'id': int(i),
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
            # For 2D, we expect UMAP 1 & UMAP 2, fallback to (0,0) if not present
            x2d = row.get('UMAP 1', 0.0)
            y2d = row.get('UMAP 2', 0.0)
            node = {
                'id': int(i),
                'protein_name': protein,
                'label': text,
                'x': float(x2d),
                'y': float(y2d),
                'group': str(label),
                'size': float(size),
                'color': color,
            }

        nodes.append(node)

    edges = []  # No edges in this scenario

    # Column options for the user
    column_options = [
        'Counts_1st_normalized',
        'Counts_2nd_normalized',
        'Counts_3rd_normalized',
        'Total_Counts_normalized',
        'Rank_normalized'
    ]
    # Also include raw columns if present
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
    info_options = ['Abstracts', 'Function [CC]']

    # Build list of CSVs for the dropdown
    csv_file_choices = get_available_csv_files()

    return render_template('index.html',
                           nodes=nodes,
                           edges=edges,
                           sel_col=sel_col,
                           sel_comp=sel_comp,
                           column_options=column_options,
                           comparison_options=comparison_options,
                           plot_type=plot_type,
                           plot_options=plot_options,
                           info_source=info_source,
                           info_options=info_options,
                           csv_file_choices=csv_file_choices,
                           selected_csv=selected_csv_filename)


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

    # We'll create a new filename for the updated version
    base_name = os.path.basename(selected_csv_filename)
    name_no_ext = os.path.splitext(base_name)[0]
    new_filename = f"{name_no_ext}_functioncc_updated.csv"
    new_filepath = os.path.join(UPDATES_FOLDER, new_filename)

    try:
        update_csv(selected_csv_filename)  # from preproc.py (in-place update)
        # Copy it into the updates folder
        df_updated = pd.read_csv(selected_csv_filename)
        df_updated.to_csv(new_filepath, index=False)
        flash(f"Function [CC] updated and saved as '{new_filename}' in updates folder.", "success")
    except Exception as e:
        flash(f"Error updating Function [CC]: {e}", "error")

    return redirect(url_for('index'))


@app.route('/fetch_abstracts', methods=['POST'])
def fetch_abstracts():
    """
    Placeholder for updating 'Abstracts'.
    Currently just fills missing rows with 'No data' to demonstrate the pattern.
    """
    global selected_csv_filename

    if not os.path.exists(selected_csv_filename):
        flash(f"Selected file {selected_csv_filename} does not exist.", "error")
        return redirect(url_for('index'))

    df = pd.read_csv(selected_csv_filename)
    if 'Abstracts' not in df.columns:
        flash("The 'Abstracts' column is missing in this CSV. Cannot fetch or update abstracts.", "error")
        return redirect(url_for('index'))

    # Placeholder logic: fill any missing with "No data"
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

