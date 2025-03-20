# preproc.py
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm

def get_data(entry_id):
    """
    Fetch 'Function [CC]' comments, 'InterPro' IDs, and PubMed IDs from UniProt XML.
    """
    uniprot_url = f"https://rest.uniprot.org/uniprotkb/{entry_id}.xml"
    try:
        response = requests.get(uniprot_url)
        response.raise_for_status()

        namespace = {"ns": "http://uniprot.org/uniprot"}
        root = ET.fromstring(response.content)

        # Extract Function [CC]
        function_comments = []
        for comment in root.findall(".//ns:comment[@type='function']", namespace):
            for text_elem in comment.findall("ns:text", namespace):
                text_value = text_elem.text.strip() if text_elem.text else "No text provided"
                function_comments.append(text_value)
        function_text = ' '.join(function_comments)

        # InterPro IDs
        interpro_ids = [
            db_ref.get("id")
            for db_ref in root.findall(".//ns:dbReference[@type='InterPro']", namespace)
            if db_ref.get("id")
        ]
        interpro_ids_str = ";".join(interpro_ids)

        # PubMed IDs
        pubmed_ids = [
            db_ref.get("id")
            for db_ref in root.findall(".//ns:dbReference[@type='PubMed']", namespace)
            if db_ref.get("id")
        ]
        pubmed_ids_str = ";".join(pubmed_ids)

        return function_text, interpro_ids_str, pubmed_ids_str

    except requests.exceptions.RequestException as e:
        return f"Error fetching data: {e}", "", ""

def update_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)

    required_columns = ["InterPro", "PubMed ID", "Function [CC]", "Abstracts"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""  # Add it if missing

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Updating CSV"):
        entry_id = row["Entry"]
        if pd.notna(entry_id):
            needs_func = not row["Function [CC]"] or pd.isna(row["Function [CC]"])
            needs_interpro = not row["InterPro"] or pd.isna(row["InterPro"])
            needs_pubmed = not row["PubMed ID"] or pd.isna(row["PubMed ID"])
            if needs_func or needs_interpro or needs_pubmed:
                function_text, interpro_ids, pubmed_ids = get_data(entry_id)
                df.at[index, "Function [CC]"] = function_text
                df.at[index, "InterPro"] = interpro_ids
                df.at[index, "PubMed ID"] = pubmed_ids

    df.to_csv(csv_file_path, index=False)
