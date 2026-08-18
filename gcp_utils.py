"""Shared Google Cloud helpers for BigQuery and GCS operations."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import bigquery, storage
from google.oauth2 import service_account
from medical_potential.config import (
    BQ_DATASET_ID,
    DIM_SCORES_TABLE,
    GCS_REPORT_BASE_PATH,
    GCS_BUCKET,
    GCS_MEDICAL_POTENTIAL_SUBFOLDER,
    GCS_PIPELINE_CACHE_BASE_PATH,
    GOOGLE_APPLICATION_CREDENTIALS,
    PROJECT_ID,
)

logger = logging.getLogger(__name__)
PILLAR_SUBFOLDER = GCS_MEDICAL_POTENTIAL_SUBFOLDER


# Client helpers
def get_bq_client() -> bigquery.Client:
    """Return an authenticated BigQuery client.

    Uses the configured service-account file when present; otherwise falls back
    to Application Default Credentials.
    """
    credentials_path = GOOGLE_APPLICATION_CREDENTIALS or "service.json"
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        return bigquery.Client(project=PROJECT_ID, credentials=credentials)
    return bigquery.Client(project=PROJECT_ID)


def get_gcs_client() -> storage.Client:
    """Return an authenticated GCS client.

    Uses the configured service-account file when present; otherwise falls back
    to Application Default Credentials.
    """
    credentials_path = GOOGLE_APPLICATION_CREDENTIALS
    if credentials_path and Path(credentials_path).exists():
        return storage.Client.from_service_account_json(credentials_path)
    return storage.Client(project=PROJECT_ID)


# Function to upload report to GCS
def upload_dimension_report_pdf_to_gcs(
    pdf_bytes: bytes,
    molecule_name: str,
    dimension_name: str,
) -> tuple[str | None, str | None]:
    """Upload a dimension report PDF to GCS and return the main and archived GCS URIs.

    Writes two copies:
    - the current report at the dimension path
    - an archived copy with a timestamp suffix
    """
    try:
        molecule = (molecule_name or "").strip()
        if not molecule:
            raise ValueError("molecule_name must be provided")

        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)

        pillar_subfolder = PILLAR_SUBFOLDER
        main_gcs_path = f"{GCS_REPORT_BASE_PATH}/{molecule}/{pillar_subfolder}/{dimension_name}.pdf"
        main_blob = bucket.blob(main_gcs_path)
        main_blob.upload_from_string(pdf_bytes, content_type="application/pdf")
        main_gcs_uri = f"gs://{GCS_BUCKET}/{main_gcs_path}"
        logger.info("[REPORT_UPLOAD] PDF uploaded to GCS for dimension '%s': %s", dimension_name, main_gcs_uri)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_gcs_path = (
            f"{GCS_REPORT_BASE_PATH}/{molecule}/{pillar_subfolder}/archived/{dimension_name}_{timestamp}.pdf"
        )
        archived_blob = bucket.blob(archived_gcs_path)
        archived_blob.upload_from_string(pdf_bytes, content_type="application/pdf")
        logger.info(
            "[REPORT_UPLOAD] Archived PDF uploaded to GCS for dimension '%s': gs://%s/%s",
            dimension_name,
            GCS_BUCKET,
            archived_gcs_path,
        )

        archive_gcs_uri = f"gs://{GCS_BUCKET}/{archived_gcs_path}"

        return main_gcs_uri, archive_gcs_uri
    except Exception as exc:
        logger.warning("[REPORT_UPLOAD] Failed to upload PDF to GCS for dimension '%s': %s", dimension_name, exc)
        return None, None


def upload_dimension_payload_cache_to_gcs(
    payload: dict,
    molecule_name: str,
    dimension_name: str,
) -> str | None:
    """Upload a final dimension payload to GCS and return the main cache URI.

    Writes two copies:
    - the current payload at the dimension path
    - an archived copy with a timestamp suffix
    """
    try:
        molecule = (molecule_name or "").strip()
        pillar = (GCS_MEDICAL_POTENTIAL_SUBFOLDER or "").strip()
        dimension = (dimension_name or "").strip()
        if not molecule:
            raise ValueError("molecule_name must be provided")
        if not pillar:
            raise ValueError("GCS_MEDICAL_POTENTIAL_SUBFOLDER must be configured")
        if not dimension:
            raise ValueError("dimension_name must be provided")

        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)

        cache_root = GCS_PIPELINE_CACHE_BASE_PATH
        main_gcs_path = f"{cache_root}/{molecule}/{pillar}/{dimension}/output_payload.json"
        payload_bytes = json.dumps(payload, indent=2, default=str).encode("utf-8")

        main_blob = bucket.blob(main_gcs_path)
        main_blob.upload_from_string(payload_bytes, content_type="application/json")
        main_gcs_uri = f"gs://{GCS_BUCKET}/{main_gcs_path}"
        logger.info(
            "[CACHE_UPLOAD] Final payload uploaded to GCS for molecule '%s', pillar '%s', dimension '%s': %s",
            molecule,
            pillar,
            dimension,
            main_gcs_uri,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_gcs_path = f"{cache_root}/{molecule}/{pillar}/{dimension}/archived/output_payload_{timestamp}.json"
        archived_blob = bucket.blob(archived_gcs_path)
        archived_blob.upload_from_string(payload_bytes, content_type="application/json")
        logger.info(
            "[CACHE_UPLOAD] Archived final payload uploaded to GCS for molecule '%s', pillar '%s', dimension '%s': gs://%s/%s",
            molecule,
            pillar,
            dimension,
            GCS_BUCKET,
            archived_gcs_path,
        )

        return main_gcs_uri
    except Exception as exc:
        logger.warning(
            "[CACHE_UPLOAD] Failed to upload final payload to GCS for molecule '%s', pillar '%s', dimension '%s': %s",
            molecule_name,
            GCS_MEDICAL_POTENTIAL_SUBFOLDER,
            dimension_name,
            exc,
        )
        return None


# Function to upload score to GBQ

## BigQuery schema
DIM_SCORES_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("product", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("pillar", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("dimension", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("score", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("rationale", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
]

def append_dimension_score_to_bigquery(
    molecule_name: str,
    dimension_name: str,
    score: float | int | None,
    pillar_name: str = "Medical Potential",
    rationale: str | None = None,
) -> None:
    """Append one dimension-score row to the configured BigQuery table.

    The payload is normalized to the shared dim-scores schema and written in
    append mode, creating the table if it does not already exist.
    """
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{DIM_SCORES_TABLE}"
    row = {
        "product": (molecule_name or "").strip() or None,
        "pillar": (pillar_name or "").strip() or None,
        "dimension": (dimension_name or "").strip() or None,
        "score": float(score) if score is not None else None,
        "rationale": (rationale or "").strip() or None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    client = get_bq_client()
    job_config = bigquery.LoadJobConfig(
        schema=DIM_SCORES_SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
    )

    load_job = client.load_table_from_json([row], table_id, job_config=job_config)
    load_job.result()
    logger.info(
        "[DIM_SCORE] Appended dimension score for molecule '%s', pillar '%s', dimension '%s' to %s",
        row["product"],
        row["pillar"],
        row["dimension"],
        table_id,
    )


