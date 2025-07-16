from __future__ import annotations
from pathlib import Path
import os
import re
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
)
from werkzeug.utils import secure_filename
from sklearn.decomposition import PCA

from embeddings import compute_embeddings
from chatbot import get_chatbot_response
from preproc import update_csv

# ──────────────────────────────
# Configuration (override via env vars)
# ──────────────────────────────
ROOT_DIR = Path(__file__).parent
SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "change_me")
UPLOAD_FOLDER: Path = Path(os.getenv("UPLOAD_FOLDER", ROOT_DIR / "uploads"))
UPDATES_FOLDER: Path = Path(os.getenv("UPDATES_FOLDER", ROOT_DIR / "updates"))
DATA_FOLDER: Path = Path(os.getenv("DATA_FOLDER", ROOT_DIR / "data"))
DEFAULT_CSV: str = os.getenv("DEFAULT_CSV", "mtuberculosis_df_abs.csv")
ALLOWED_EXTENSIONS = {"csv"}

# UI select‑box defaults / options
COLUMN_DEFAULT = "Total_Counts"
COMPARISON_DEFAULT = "All proteomes"
PLOT_DEFAULT = "2D UMAP Based"
INFO_DEFAULT = "Function [CC]" 

COMPARISON_OPTIONS = [
    "All proteomes",
    "Mycobacterium tuberculosis",
    "vs smegmatis",
    "vs marinum",
    "vs leprae",
    "vs kansasii",
    "vs intracellulare",
    "vs fortuitum",
    "vs bovis",
]
PLOT_OPTIONS = ["2D UMAP Based", "3D PCA Based"]
INFO_OPTIONS = ["Function [CC]","Abstracts"]

# Ensure folders exist
for folder in (UPLOAD_FOLDER, UPDATES_FOLDER, DATA_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────
# Flask app setup
# ──────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Single dict to store mutable UI state (avoids many module‑level globals)
state: Dict[str, Any] = {
    "csv_path": DATA_FOLDER / DEFAULT_CSV,
    "sel_col": COLUMN_DEFAULT,
    "sel_comp": COMPARISON_DEFAULT,
    "plot_type": PLOT_DEFAULT,
    "info_source": INFO_DEFAULT,
}

# ──────────────────────────────
# Helper functions
# ──────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_csv_choices() -> List[Tuple[str, Path]]:
    """Return [[label, full_path], …] for dropdown."""
    choices: List[Tuple[str, Path]] = []
    # Default
    default_path = DATA_FOLDER / DEFAULT_CSV
    if default_path.exists():
        choices.append((f"Default: {DEFAULT_CSV}", default_path))
    # Uploaded & updated
    for folder, tag in ((UPLOAD_FOLDER, "Uploaded"), (UPDATES_FOLDER, "Updated")):
        for f in folder.glob("*.csv"):
            choices.append((f"[{tag}] {f.name}", f))
    return choices


def load_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        flash(f"Error reading CSV '{csv_path.name}': {exc}", "error")
        return pd.DataFrame()


def clean_text(text: str) -> str:
    text = re.sub(r"\d+", "", str(text))
    # remove bracketed content recursively
    while "(" in text and ")" in text:
        text = re.sub(r"\([^()]*\)", "", text)
    return text.strip()


def normalize_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        rng = df[c].max() - df[c].min()
        df[c + "_normalized"] = 0 if rng == 0 else (df[c] - df[c].min()) / rng
    return df


# ──────────────────────────────
# Routes
# ──────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    # 1️⃣ Handle uploads & form data
    if request.method == "POST":
        # Upload
        if (file := request.files.get("file")) and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            dest = UPLOAD_FOLDER / filename
            file.save(dest)
            state["csv_path"] = dest
            flash(f"File '{filename}' uploaded successfully!", "success")
        # Dropdown change
        if sel := request.form.get("selected_csv"):
            state["csv_path"] = Path(sel)
        # UI controls
        state["sel_col"] = request.form.get("sel_col", state["sel_col"])
        state["sel_comp"] = request.form.get("sel_comp", state["sel_comp"])
        state["plot_type"] = request.form.get("plot_type", state["plot_type"])
        state["info_source"] = request.form.get("info_source", state["info_source"])
        return redirect(url_for("index"))

    # 2️⃣ Load & validate CSV
    df = load_csv(state["csv_path"])
    if df.empty:
        return render_template("index.html")

    if state["info_source"] not in df.columns:
        flash(f"Column '{state['info_source']}' missing in CSV.", "error")

    # Pre‑processing
    if "Organism" in df.columns:
        df["Organism"] = df["Organism"].map(clean_text)
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        mapping = {org: palette[i % len(palette)] for i, org in enumerate(df["Organism"].unique())}
        df["Color"] = df["Organism"].map(mapping)
    else:
        df["Color"] = "#1f77b4"

    # Counts normalisation (specific to Mycobacterium tuberculosis project)
    counts_xlsx = DATA_FOLDER / "counts_all_stages_MAGECK_with_ES.xlsx"
    if counts_xlsx.exists():
        counts_df = pd.read_excel(counts_xlsx)
        counts_df["clean_orf"] = counts_df["orf"].str.replace(r"(?<=RV)BD", "", regex=True)
        counts_df["clean_name"] = counts_df["name"].str.replace(r"(?<=RV)BD", "", regex=True)
        tb_df = (
            df if "Organism" not in df.columns else df[df["Organism"] == "Mycobacterium tuberculosis"].copy()
        )
        cols = ["Counts_1st", "Counts_2nd", "Counts_3rd", "Total_Counts", "Rank"]
        for c in cols:
            if c not in tb_df.columns:
                tb_df[c] = np.nan
        for _, row in counts_df.iterrows():
            pattern = f"{re.escape(str(row['clean_orf']))}|{re.escape(str(row['clean_name']))}"
            mask = tb_df.get("Gene Names", pd.Series(dtype=bool)).str.contains(pattern, case=False, na=False)
            if mask.any():
                tb_df.loc[mask, cols] = [row[c] for c in cols]
        tb_df = normalize_columns(tb_df, cols)
    else:
        tb_df = pd.DataFrame()

    # Build plot_df based on comparison choice
    sel_col = state["sel_col"]
    if tb_df.empty or sel_col not in tb_df.columns:
        sel_col = state["sel_col"] = f"{COLUMN_DEFAULT}_normalized"
    if state["sel_comp"] == "All proteomes":
        plot_df = df.merge(tb_df[["Entry", sel_col, "Rank"]], on="Entry", how="left").fillna({sel_col: 0})
    elif state["sel_comp"] == "Mycobacterium tuberculosis":
        plot_df = tb_df.copy()
    else:
        other = state["sel_comp"].split("vs ")[-1]
        subset = df if "Organism" not in df.columns else df[
            (df["Organism"] == "Mycobacterium tuberculosis")
            | df["Organism"].str.contains(other, case=False, na=False)
        ]
        plot_df = subset.merge(tb_df[["Entry", sel_col, "Rank"]], on="Entry", how="left").fillna({sel_col: 0})

    # Node size scaling
    min_sz, max_sz = 10, 50
    rng = plot_df[sel_col].max() - plot_df[sel_col].min()
    norm = 0 if rng == 0 else (plot_df[sel_col] - plot_df[sel_col].min()) / rng
    plot_df["Size"] = min_sz + norm * (max_sz - min_sz)

    # Embeddings / coordinates
    embeddings = None
    if state["info_source"] in df.columns:
        embeddings = compute_embeddings(column=state["info_source"])
    coords = None
    if embeddings is not None and state["plot_type"].startswith("3D"):
        cache_file = DATA_FOLDER / f"coordinates_{state['info_source']}.npy"
        coords = np.load(cache_file) if cache_file.exists() else PCA(3).fit_transform(embeddings)
        if not cache_file.exists():
            np.save(cache_file, coords)

    # Build nodes list (2D or 3D)
    nodes: List[Dict[str, Any]] = []
    for idx, row in plot_df.iterrows():
        coords_tuple = (
            (row.get("UMAP 1", 0.0), row.get("UMAP 2", 0.0), None)
            if coords is None
            else (*coords[idx],)
        )
        nodes.append(
            {
                "id": int(idx),
                "protein_name": row.get("Protein names", "N/A"),
                "label": "<br>".join(
                    [
                        f"Protein: {row.get('Protein names', 'N/A')}",
                        f"Organism: {row.get('Organism', 'N/A')}",
                        f"Gene: {row.get('Gene Names', 'N/A')}",
                        f"Pathway: {row.get('Pathway', 'N/A')}",
                        f"{sel_col}: {row.get(sel_col, 'N/A')}",
                        f"Annotation: {row.get('Annotation', 'N/A')}",
                    ]
                ),
                "x": float(coords_tuple[0]),
                "y": float(coords_tuple[1]),
                "z": float(coords_tuple[2]) if coords_tuple[2] is not None else None,
                "group": str(row.get("Cluster Label", "N/A")),
                "size": float(row["Size"]),
                "color": row["Color"],
            }
        )

    context = dict(
        nodes=nodes,
        edges=[],  # None for now
        sel_col=sel_col,
        sel_comp=state["sel_comp"],
        column_options=[c for c in plot_df.columns if c.endswith("_normalized")] + [COLUMN_DEFAULT],
        comparison_options=COMPARISON_OPTIONS,
        plot_type=state["plot_type"],
        plot_options=PLOT_OPTIONS,
        info_source=state["info_source"],
        info_options=INFO_OPTIONS,
        csv_file_choices=[(label, str(p)) for label, p in get_csv_choices()],
        selected_csv=str(state["csv_path"]),
    )
    return render_template("index.html", **context)


@app.route("/chatbot", methods=["POST"])
def chatbot():
    payload = request.get_json()
    response = get_chatbot_response(
        node_ids=payload.get("node_ids", []),
        message=payload.get("message", ""),
        include_similar=payload.get("include_similar", True),
        info_source=state["info_source"],
        csv_path=str(state["csv_path"]),
    )
    # info_source = info_source
    # print(info_source)
    return jsonify({"message": response})


@app.route("/fetch_function_cc", methods=["POST"])
def fetch_function_cc():
    try:
        update_csv(str(state["csv_path"]))
        new_file = UPDATES_FOLDER / (state["csv_path"].stem + "_functioncc_updated.csv")
        pd.read_csv(state["csv_path"]).to_csv(new_file, index=False)
        flash(f"Function [CC] updated → {new_file.name}", "success")
    except Exception as exc:
        flash(f"Error updating Function [CC]: {exc}", "error")
    return redirect(url_for("index"))


@app.route("/fetch_abstracts", methods=["POST"])
def fetch_abstracts():
    try:
        df = pd.read_csv(state["csv_path"])
        if "Abstracts" not in df.columns:
            flash("'Abstracts' column missing – cannot update abstracts.", "error")
        else:
            df["Abstracts"].fillna("No data", inplace=True)
            new_file = UPDATES_FOLDER / (state["csv_path"].stem + "_abstracts_updated.csv")
            df.to_csv(new_file, index=False)
            flash(f"Abstracts updated → {new_file.name}", "success")
    except Exception as exc:
        flash(f"Error updating Abstracts: {exc}", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Port 0 asks the OS for an available port – useful during dev / behind
    # reverse proxy setups.
    app.run(debug=True, port=0)
