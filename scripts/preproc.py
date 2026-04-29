# preproc.py
import logging
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_data(entry_id):
    """
    Fetch 'Function [CC]' comments, 'InterPro' IDs, and PubMed IDs from UniProt XML.

    Raises:
        requests.exceptions.RequestException: On any network or HTTP error.
        ValueError: If the response body is not valid XML.
    """
    uniprot_url = f"https://rest.uniprot.org/uniprotkb/{entry_id}.xml"

    # Allow RequestException to propagate — no catching here.
    response = requests.get(uniprot_url)
    response.raise_for_status()

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        raise ValueError(f"Malformed XML response for entry {entry_id}: {e}") from e

    namespace = {"ns": "http://uniprot.org/uniprot"}

    # Extract Function [CC]
    function_comments = []
    for comment in root.findall(".//ns:comment[@type='function']", namespace):
        for text_elem in comment.findall("ns:text", namespace):
            text_value = text_elem.text.strip() if text_elem.text else "No text provided"
            function_comments.append(text_value)
    function_text = " ".join(function_comments)

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


def fetch_with_retry(entry_id, max_retries=3):
    """
    Call get_data with exponential backoff on transient network errors.

    Raises:
        requests.exceptions.RequestException: After all retries are exhausted.
        ValueError: Immediately on malformed XML (not retried).
    """
    for attempt in range(max_retries):
        try:
            return get_data(entry_id)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                f"Retry {attempt + 1}/{max_retries} for {entry_id} after {wait}s: {e}"
            )
            time.sleep(wait)


def update_csv(csv_file_path):
    """
    Update the CSV at csv_file_path with Function [CC], InterPro, and PubMed ID
    data fetched from UniProt for each entry that is missing any of those fields.

    Progress is saved every 100 rows. Rows that fail after all retries are skipped
    with a warning rather than crashing the entire run.
    """
    df = pd.read_csv(csv_file_path)

    required_columns = ["InterPro", "PubMed ID", "Function [CC]", "Abstracts"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    skipped = 0
    updated = 0

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Updating CSV"):
        entry_id = row["Entry"]
        if pd.isna(entry_id):
            continue

        needs_func = not row["Function [CC]"] or pd.isna(row["Function [CC]"])
        needs_interpro = not row["InterPro"] or pd.isna(row["InterPro"])
        needs_pubmed = not row["PubMed ID"] or pd.isna(row["PubMed ID"])

        if not (needs_func or needs_interpro or needs_pubmed):
            continue

        logger.debug(f"Processing entry {entry_id} (row {index})")

        try:
            function_text, interpro_ids, pubmed_ids = fetch_with_retry(entry_id)
        except Exception as e:
            logger.warning(f"Skipping {entry_id}: {e}")
            skipped += 1
        else:
            df.at[index, "Function [CC]"] = function_text
            df.at[index, "InterPro"] = interpro_ids
            df.at[index, "PubMed ID"] = pubmed_ids
            updated += 1

        # Rate-limit UniProt API calls.
        time.sleep(0.5)

        # Periodic save to preserve progress on interruption.
        if index % 100 == 0 and index > 0:
            df.to_csv(csv_file_path, index=False)
            logger.info(f"Progress saved at row {index}")

    df.to_csv(csv_file_path, index=False)
    logger.info(
        f"update_csv complete: {updated} rows updated, {skipped} rows skipped."
    )
