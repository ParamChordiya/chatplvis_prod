import pandas as pd
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm

def get_data(entry_id):
    """
    Fetch 'Function [CC]' comments, 'InterPro' IDs, and PubMed IDs from the UniProt XML API for a given ENTRY ID.

    Parameters:
        entry_id (str): The UniProt ENTRY ID.

    Returns:
        tuple: Concatenated 'text' values from Function [CC] comments, a list of 'InterPro' IDs, and a list of PubMed IDs.
    """
    uniprot_url = f"https://rest.uniprot.org/uniprotkb/{entry_id}.xml"

    try:
        # Send GET request to UniProt API
        response = requests.get(uniprot_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the XML response
        namespace = {"ns": "http://uniprot.org/uniprot"}  # Define the namespace
        root = ET.fromstring(response.content)

        # Extract all Function [CC] comments
        function_comments = []
        for comment in root.findall(".//ns:comment[@type='function']", namespace):
            for text_elem in comment.findall("ns:text", namespace):
                text_value = text_elem.text.strip() if text_elem.text else "No text provided"
                function_comments.append(text_value)

        # Concatenate all text values
        function_text = ' '.join(function_comments)

        # Extract all InterPro IDs
        interpro_ids = [
            db_ref.get("id")
            for db_ref in root.findall(".//ns:dbReference[@type='InterPro']", namespace)
            if db_ref.get("id")
        ]
        interpro_ids_str = ";".join(interpro_ids)  # Convert list to semi-colon-separated string

        # Extract all PubMed IDs
        pubmed_ids = [
            db_ref.get("id")
            for db_ref in root.findall(".//ns:dbReference[@type='PubMed']", namespace)
            if db_ref.get("id")
        ]
        pubmed_ids_str = ";".join(pubmed_ids)  # Convert list to semi-colon-separated string

        return function_text, interpro_ids_str, pubmed_ids_str
    except requests.exceptions.RequestException as e:
        return f"Error fetching data: {e}", "", ""


def update_csv(csv_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Ensure required columns exist
    required_columns = ["InterPro", "PubMed ID", "Function [CC]", "Abstracts"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""  # Add column if it doesn't exist

    # Iterate over each entry and fetch data only if missing
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Updating CSV"):
        entry_id = row["Entry"]
        # Check if data is missing for this row
        if pd.notna(entry_id) and (pd.isna(row["Function [CC]"]) or pd.isna(row["InterPro"]) or pd.isna(row["PubMed ID"]) or not row["Function [CC]"]):
            function_text, interpro_ids, pubmed_ids = get_data(entry_id)
            df.at[index, "Function [CC]"] = function_text
            df.at[index, "InterPro"] = interpro_ids
            df.at[index, "PubMed ID"] = pubmed_ids

    # Optionally save the updated CSV
    df.to_csv(csv_file_path, index=False)