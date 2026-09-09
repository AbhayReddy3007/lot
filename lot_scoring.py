"""SoC → Lines-of-Therapy (LOT) scoring — core logic.

Reads country-level Standard-of-Care (SoC) PDFs from GCS (one subfolder per
country under ``GCS_SOC_BASE_PATH``), looks up each drug's Mechanism of
Action (MOA) from the BigQuery MOA lookup table, asks Gemini to (1) extract a
per-country SoC LOT benchmark and (2) classify each supplied drug against
that benchmark, then pushes one row per drug/country combination to
BigQuery:

    drug_name, country, lot_score, lot_type, rationale, confidence, final_lot_score

final_lot_score is a per-drug aggregate across all countries, NOT a per-row
value: (US lot_score x US_WEIGHT) + OTHER_COUNTRY_WEIGHT x sum(lot_score for
every other country). It is repeated on every row for that drug.

SoC benchmark extraction (the PDF-read + Gemini-extraction step) is cached to
GCS as JSON under ``gs://GCS_BUCKET/{LOT_BENCHMARK_PATH}/``, one blob per
country. On subsequent runs, if a cached benchmark blob already exists for a
country, the PDFs are not re-read and the benchmark is not re-generated — the
cached benchmark text is reused directly for the overlay analysis.

The entry point (``main()``) lives in ``line_of_treatment.py``, which imports
everything it needs from this module.

Place these modules at ``medical_potential/lot_scoring.py`` and
``medical_potential/line_of_treatment.py`` so they can reuse
``medical_potential.gcp_utils`` and ``medical_potential.config`` exactly like
the rest of the package.

NEW CONFIG REQUIRED
--------------------
Add the following to ``medical_potential/config.py`` (they do not exist yet):

    MOA_LOOKUP_TABLE            # e.g. "brands" — BQ table (under PROJECT_ID /
                                #   BQ_DATASET_ID) with Cleaned_Generic_Name /
                                #   Mechanism_of_Action / Mechanism_of_Action_Detailed
    GCS_SOC_BASE_PATH           # e.g. "SOC"    — GCS prefix under GCS_BUCKET that
                                #   contains one subfolder per country, each holding
                                #   that country's SoC PDF(s)
    GEMINI_FLASH_PREVIEW_MODEL  # e.g. "gemini-2.5-flash" — Gemini model used for
                                #   both SoC extraction and overlay analysis
    US_WEIGHT                   # e.g. 0.58 — weight applied to the US lot_score
    OTHER_COUNTRY_WEIGHT        # e.g. 0.14 — weight applied to each non-US lot_score
    LOT_TABLE                   # BQ table results are pushed to, e.g. "soc_lot_scores"
    LOT_BENCHMARK_PATH          # GCS prefix (under GCS_BUCKET) where per-country
                                #   SoC benchmark JSON cache blobs are read from /
                                #   written to, e.g. "lot_benchmarks"

PROJECT_ID and BQ_DATASET_ID are reused as-is (both for the MOA lookup table
and for the LOT results table).

GEMINI_API_KEY is loaded from the environment (.env), matching the existing
scripts (U2.py / lot.py) rather than from config.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from google.cloud import bigquery
from google.genai import types

from medical_potential.config import (
    BQ_DATASET_ID,
    GCS_BUCKET,
    GCS_SOC_BASE_PATH,
    GEMINI_FLASH_PREVIEW_MODEL,
    LOT_BENCHMARK_PATH,
    LOT_TABLE,
    MOA_LOOKUP_TABLE,
    OTHER_COUNTRY_WEIGHT,
    PROJECT_ID,
    US_WEIGHT,
)
from medical_potential.gcp_utils import get_bq_client, get_gcs_client

logger = logging.getLogger(__name__)

# ==============================
# ENV / GEMINI CLIENT
# ==============================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found.")
client = genai.Client(api_key=GEMINI_API_KEY)

# ==============================
# CONFIG / CONSTANTS
# ==============================
# LOT type -> numeric score
LOT_SCORE_MAP: dict[str, int] = {
    "first-line standard of care": 5,
    "strong first-line alternative / dominant second-line": 4,
    "second-line option": 3,
    "third-line or restricted niche use": 2,
    "salvage / last-resort use": 1,
}

# Fallback keyword matching, checked in order, for LOT strings that don't
# exactly match LOT_SCORE_MAP (e.g. minor wording drift from the LLM).
LOT_KEYWORD_FALLBACK: list[tuple[str, int]] = [
    ("salvage", 1),
    ("last-resort", 1),
    ("third-line", 2),
    ("restricted niche", 2),
    ("strong first-line alternative", 4),
    ("dominant second-line", 4),
    ("second-line", 3),
    ("first-line", 5),
]

US_ALIASES = {"us", "usa", "u.s.", "u.s.a.", "united states", "united_states", "united states of america"}


# ==============================
# CLI ARGUMENTS
# ==============================
def parse_drugs() -> list[str]:
    """Reads drug names from CLI args, falling back to an interactive prompt."""
    if len(sys.argv) > 1:
        drugs = sys.argv[1:]
    else:
        user_input = input("No drugs supplied. Enter drug names separated by spaces: ").strip()
        if not user_input:
            raise SystemExit("No drug names provided. Exiting.")
        drugs = user_input.split()
    logger.info("[SOC_LOT] Drugs to analyse: %s", ", ".join(drugs))
    return drugs


# ==============================
# BIGQUERY: MOA LOOKUP
# ==============================
def lookup_moa(drug_names: list[str]) -> dict[str, dict]:
    """Maps each drug name to its MOA via BigQuery. Falls back to 'Unknown'."""
    bq_client = get_bq_client()
    table_ref = f"`{PROJECT_ID}.{BQ_DATASET_ID}.{MOA_LOOKUP_TABLE}`"

    placeholders = ", ".join(f"@drug_{i}" for i in range(len(drug_names)))
    query = f"""
        SELECT
            Cleaned_Generic_Name,
            Mechanism_of_Action,
            Mechanism_of_Action_Detailed
        FROM {table_ref}
        WHERE LOWER(Cleaned_Generic_Name) IN ({placeholders})
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(f"drug_{i}", "STRING", name.lower())
            for i, name in enumerate(drug_names)
        ]
    )
    results = bq_client.query(query, job_config=job_config).result()

    moa_map_lower: dict[str, dict] = {}
    for row in results:
        moa_map_lower[row.Cleaned_Generic_Name.lower()] = {
            "moa": row.Mechanism_of_Action or "Unknown",
            "moa_detailed": row.Mechanism_of_Action_Detailed or "",
        }

    drug_moa: dict[str, dict] = {}
    for drug in drug_names:
        match = moa_map_lower.get(drug.lower())
        if match:
            drug_moa[drug] = match
        else:
            logger.warning("[SOC_LOT] '%s' not found in BigQuery table. Using 'Unknown' as MOA.", drug)
            drug_moa[drug] = {"moa": "Unknown", "moa_detailed": ""}

    return drug_moa


# ==============================
# GCS: DISCOVER + READ SoC PDFs
# ==============================
def discover_countries_gcs(base_path: str = GCS_SOC_BASE_PATH) -> dict[str, list[str]]:
    """Lists SoC PDFs in GCS, grouped by country.

    Expects blobs laid out as: {base_path}/{Country}/{...}.pdf
    Returns {country_name: [blob_name, ...]}.
    """
    gcs_client = get_gcs_client()
    bucket = gcs_client.bucket(GCS_BUCKET)
    prefix = f"{base_path.rstrip('/')}/"

    countries: dict[str, list[str]] = {}
    for blob in gcs_client.list_blobs(bucket, prefix=prefix):
        if not blob.name.lower().endswith(".pdf"):
            continue
        relative = blob.name[len(prefix):]
        parts = relative.split("/")
        if len(parts) < 2:
            # No country subfolder — skip stray files directly under base_path
            continue
        country = parts[0]
        countries.setdefault(country, []).append(blob.name)

    if not countries:
        raise FileNotFoundError(f"No country PDFs found under gs://{GCS_BUCKET}/{prefix}")

    for country in countries:
        countries[country].sort()

    return countries


def extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return "".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def extract_country_pdfs_text_gcs(blob_names: list[str]) -> str:
    """Downloads and concatenates text from all PDFs for a single country."""
    gcs_client = get_gcs_client()
    bucket = gcs_client.bucket(GCS_BUCKET)

    sections = []
    for i, blob_name in enumerate(blob_names, 1):
        blob = bucket.blob(blob_name)
        pdf_bytes = blob.download_as_bytes()
        text = extract_pdf_text_from_bytes(pdf_bytes)
        filename = blob_name.rsplit("/", 1)[-1]
        sections.append(f"--- Document {i}: {filename} ---\n{text}")
    return "\n\n".join(sections)


# ==============================
# BENCHMARK CACHE (GCS JSON, one blob per country)
# ==============================
def _slugify_country(country: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", country.strip().lower())
    return slug.strip("_") or "unknown_country"


def get_benchmark_cache_blob_name(country: str) -> str:
    """Returns the GCS blob name (path within GCS_BUCKET) for a country's cached benchmark."""
    base = LOT_BENCHMARK_PATH.strip("/")
    return f"{base}/{_slugify_country(country)}.json"


def load_cached_benchmark(country: str) -> str | None:
    """Returns the cached SoC benchmark text for a country, or None if absent.

    Reads from gs://GCS_BUCKET/{LOT_BENCHMARK_PATH}/{country_slug}.json.
    """
    blob_name = get_benchmark_cache_blob_name(country)
    gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
    try:
        gcs_client = get_gcs_client()
        bucket = gcs_client.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            return None
        payload = json.loads(blob.download_as_text())
        soc_output = payload.get("soc_output")
        if soc_output:
            logger.info("[SOC_LOT] Using cached benchmark for '%s': %s", country, gcs_uri)
            return soc_output
    except Exception:
        logger.exception("[SOC_LOT] Failed to read cached benchmark for '%s' at %s", country, gcs_uri)
    return None


def save_benchmark_to_cache(country: str, soc_output: str) -> None:
    """Writes the SoC benchmark text for a country to the GCS JSON cache.

    Writes to gs://GCS_BUCKET/{LOT_BENCHMARK_PATH}/{country_slug}.json.
    """
    blob_name = get_benchmark_cache_blob_name(country)
    gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
    payload = {
        "country": country,
        "soc_output": soc_output,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        gcs_client = get_gcs_client()
        bucket = gcs_client.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(payload, indent=2, ensure_ascii=False),
            content_type="application/json",
        )
        logger.info("[SOC_LOT] Cached benchmark for '%s' at %s", country, gcs_uri)
    except Exception:
        logger.exception("[SOC_LOT] Failed to cache benchmark for '%s' at %s", country, gcs_uri)


# ==============================
# SYSTEM INSTRUCTION (SoC extraction)
# ==============================
SYSTEM_INSTRUCTION = """You are a Clinical Pharmacist and Diabetes/Obesity Treatment Pathway Analyst.
Analyze the provided Standard of Care (SoC) document(s) and create a country-level Lines of Therapy (LOT) benchmark.

LANGUAGE HANDLING:
- The source documents may be in English, French, or a mix of languages.
- Regardless of the source language, ALL output must be in English.
- Translate any non-English clinical terms, drug names, and therapy descriptions into their standard English equivalents.

PURPOSE:
Neutral, evidence-based extraction for downstream LOT comparison.

SCOPE:
- Type 2 Diabetes Mellitus (T2DM)
- Obesity / Weight Management

NOTE:
- You may receive text extracted from multiple SoC PDFs for the same country.
- Synthesise all documents into a single unified LOT benchmark for that country.
- If documents conflict, prefer the most recent or most authoritative guidance.

OUTPUT FORMAT STRICT:
Country: [Country]
Country-Level SoC LOT Benchmark

1L:
Recommended therapies/interventions:
Patient segment or trigger:
Treatment goal or rationale:

2L:
Recommended therapies/interventions:
Patient segment or trigger:
Treatment goal or rationale:

3L:
Recommended therapies/interventions:
Patient segment or trigger:
Treatment goal or rationale:

Salvage:
Recommended therapies/interventions:
Patient segment or trigger:
Treatment goal or rationale:

Therapy Classes Explicitly Mentioned:
- List all major classes

FORMATTING RULES:
- Output plain business English.
- Do NOT use markdown.
- Do NOT use LaTeX.
- Use Unicode symbols directly: ≥ ≤ kg/m2.
- Do NOT use *, **, #, or $.
- Produce plain text."""


# ==============================
# OVERLAY PROMPT (LOT classification + confidence)
# ==============================
def build_overlay_prompt(drugs: list[str], drug_moa: dict[str, dict]) -> str:
    molecule_lines = []
    for i, drug in enumerate(drugs):
        moa = drug_moa[drug]["moa"]
        moa_det = drug_moa[drug]["moa_detailed"]
        detail = f" ({moa_det})" if moa_det else ""
        molecule_lines.append(f"{i + 1}. {drug} - MOA: {moa}{detail}")
    molecule_list = "\n".join(molecule_lines)

    output_blocks = "\n\n".join(
        f"{drug}:\nMolecule: {drug}\nMOA: {drug_moa[drug]['moa']}\n"
        f"Final LoT Category:\nRationale:\nConfidence:"
        for drug in drugs
    )

    return f"""You are a Senior Market Access Analyst.

Task:
Determine the most appropriate Line of Treatment (LOT) classification for each molecule below based ONLY on the supplied MOA and the SoC definitions provided above.

Molecules and their MOA:
{molecule_list}

LOT ASSIGNMENT RULES:
- Identify the pharmacologic class, therapeutic modality, or treatment approach from the MOA.
- Match the MOA to the therapy classes, mechanisms, modalities, or treatment approaches explicitly described in the SoC.
- Determine the earliest treatment line in which the matching therapy class, mechanism, modality, or treatment approach appears.
- If the MOA aligns with multiple treatment lines, assign the earliest applicable line.
- Do NOT use historical treatment sequencing, current prescribing patterns, external guidelines, prior knowledge, or assumptions outside the supplied SoC.
- Evaluate the MOA against all SoC pathways and patient segments described in the benchmark.
- When a therapy class, mechanism, modality, or treatment approach appears in multiple SoC pathways, prioritize the earliest treatment line in which it is explicitly recommended.
- Only assign Second-Line when the MOA aligns primarily with therapies described as add-on, substitute, escalation, replacement, or post-failure options relative to First-Line treatment.
- Only assign Third-Line when the MOA aligns primarily with therapies described as later-line escalation, refractory-disease management, restricted-use therapies, or options used after failure of earlier treatment lines.
- Only assign Salvage when the MOA aligns primarily with rescue therapies, transplantation, last-resort interventions, or therapies explicitly described in the Salvage section of the SoC.
- When uncertainty exists, assign the earliest treatment line supported by the MOA-to-SoC mapping.

CLASSIFICATION OPTIONS (use this exact wording for Final LoT Category):
- First-line standard of care
- Strong first-line alternative / dominant second-line
- Second-line option
- Third-line or restricted niche use
- Salvage / last-resort use

CONFIDENCE:
- For each molecule, also output a Confidence score from 0 to 100 reflecting how directly the MOA maps to explicit language in the SoC.
- Use a high confidence (80-100) only when the SoC explicitly names the therapy class/mechanism at a specific line.
- Use a medium confidence (50-79) when the mapping is inferred from a closely related class or modality.
- Use a low confidence (0-49) when the SoC does not clearly address this MOA and the line was inferred indirectly.
- Output Confidence as a bare integer (e.g. "82"), with no % sign or extra words.

OUTPUT FORMAT (repeat exactly for every molecule, keep field labels verbatim):
{output_blocks}

FORMATTING RULES:
- Output plain business English.
- Do NOT use markdown.
- Do NOT use LaTeX.
- Use Unicode symbols directly.
- Produce plain text."""


# ==============================
# TEXT CLEANUP
# ==============================
def clean_text(text: str) -> str:
    replacements = {
        "$\\ge$": "≥",
        "$\\le$": "≤",
        "\\ge": "≥",
        "\\le": "≤",
        "**": "",
        "*": "",
        "#": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\$+", "", text)
    return text.strip()


# ==============================
# GEMINI CALLS
# ==============================
def run_soc_extraction(pdf_text: str, country: str) -> str:
    response = client.models.generate_content(
        model=GEMINI_FLASH_PREVIEW_MODEL,
        contents=(f"Country: {country}\n\nBelow are the full SoC document(s) for {country}:\n\n{pdf_text}"),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, temperature=0.1),
    )
    return response.text if response.text else "No response received."


def run_overlay_analysis(soc_text: str, overlay_prompt: str) -> str:
    response = client.models.generate_content(
        model=GEMINI_FLASH_PREVIEW_MODEL,
        contents=(
            "Below is the extracted SoC benchmark. "
            "Use it as the sole reference for LOT assignment.\n\n"
            f"{soc_text}\n\n{overlay_prompt}"
        ),
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text if response.text else "No response received."


# ==============================
# PARSE OVERLAY OUTPUT -> STRUCTURED FIELDS
# ==============================
def split_drug_blocks(text: str, drugs: list[str]) -> dict[str, str]:
    """Splits the overlay LLM output into one text block per drug heading."""
    positions: list[tuple[int, str]] = []
    for drug in drugs:
        pattern = re.compile(rf"(?im)^{re.escape(drug)}\s*:\s*$")
        m = pattern.search(text)
        if m:
            positions.append((m.start(), drug))

    positions.sort(key=lambda p: p[0])
    blocks: dict[str, str] = {}
    for i, (start, drug) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        blocks[drug] = text[start:end]
    return blocks


def extract_field(block: str, label: str) -> str:
    pattern = re.compile(
        rf"(?im)^{re.escape(label)}\s*:\s*(.*?)(?=\n[A-Za-z][A-Za-z /\\-]*:\s*|\Z)",
        re.DOTALL,
    )
    m = pattern.search(block)
    return clean_text(m.group(1)) if m else ""


def parse_overlay_output(overlay_text: str, drugs: list[str]) -> dict[str, dict]:
    """Returns {drug: {"lot_type": ..., "rationale": ..., "confidence": int|None}}."""
    blocks = split_drug_blocks(overlay_text, drugs)
    parsed: dict[str, dict] = {}
    for drug in drugs:
        block = blocks.get(drug, "")
        if not block:
            logger.warning("[SOC_LOT] Could not locate output block for drug '%s'", drug)
        lot_type = extract_field(block, "Final LoT Category")
        rationale = extract_field(block, "Rationale")
        confidence_raw = extract_field(block, "Confidence")

        confidence: int | None = None
        conf_match = re.search(r"\d{1,3}", confidence_raw)
        if conf_match:
            confidence = max(0, min(100, int(conf_match.group(0))))

        parsed[drug] = {"lot_type": lot_type, "rationale": rationale, "confidence": confidence}
    return parsed


# ==============================
# LOT SCORE MAPPING
# ==============================
def map_lot_type_to_score(lot_type: str) -> int | None:
    if not lot_type:
        return None
    normalized = lot_type.strip().lower()
    if normalized in LOT_SCORE_MAP:
        return LOT_SCORE_MAP[normalized]
    for keyword, score in LOT_KEYWORD_FALLBACK:
        if keyword in normalized:
            return score
    logger.warning("[SOC_LOT] Unrecognized LOT type '%s' — no score assigned", lot_type)
    return None


def is_us(country: str) -> bool:
    normalized = re.sub(r"[_\-]+", " ", country).strip().lower()
    return normalized in US_ALIASES


def compute_final_lot_scores_per_drug(rows: list["LotRow"]) -> dict[str, float]:
    """Computes one aggregate final_lot_score per drug across all countries.

    final_lot_score = (US lot_score x US_WEIGHT) + OTHER_COUNTRY_WEIGHT x sum(
    lot_score for every other country). Countries with no resolvable
    lot_score are skipped.
    """
    scores_by_drug: dict[str, dict[str, list[int]]] = {}
    for r in rows:
        if r.lot_score is None:
            continue
        bucket = scores_by_drug.setdefault(r.drug_name, {"us": [], "other": []})
        bucket["us" if is_us(r.country) else "other"].append(r.lot_score)

    final_scores: dict[str, float] = {}
    for drug, buckets in scores_by_drug.items():
        us_score = sum(buckets["us"]) * US_WEIGHT  # normally 0 or 1 US row
        other_score = sum(buckets["other"]) * OTHER_COUNTRY_WEIGHT
        final_scores[drug] = round(us_score + other_score, 4)
    return final_scores


# ==============================
# PER-COUNTRY PROCESSING
# ==============================
@dataclass
class LotRow:
    drug_name: str
    country: str
    lot_score: int | None
    lot_type: str
    rationale: str
    confidence: int | None
    final_lot_score: float | None


def get_or_build_soc_benchmark(country: str, blob_names: list[str]) -> str:
    """Returns the SoC benchmark text for a country, using the local cache when present.

    Only reads PDFs from GCS and calls Gemini for extraction when no cached
    benchmark JSON blob exists yet for this country under
    gs://GCS_BUCKET/{LOT_BENCHMARK_PATH}/.
    """
    cached = load_cached_benchmark(country)
    if cached is not None:
        return cached

    n_pdfs = len(blob_names)
    logger.info("[SOC_LOT] Reading %d PDF(s) from GCS for '%s'...", n_pdfs, country)
    pdf_text = extract_country_pdfs_text_gcs(blob_names)

    logger.info("[SOC_LOT] Extracting SoC benchmark for '%s'...", country)
    soc_output = run_soc_extraction(pdf_text, country)

    save_benchmark_to_cache(country, soc_output)
    return soc_output


def process_country(country: str, blob_names: list[str], drugs: list[str], overlay_prompt: str) -> list[LotRow]:
    n_pdfs = len(blob_names)
    logger.info("[SOC_LOT] Processing: %s (%d PDF%s)", country, n_pdfs, "s" if n_pdfs != 1 else "")

    soc_output = get_or_build_soc_benchmark(country, blob_names)

    logger.info("[SOC_LOT] Running overlay analysis for '%s'...", country)
    overlay_output = run_overlay_analysis(soc_output, overlay_prompt)

    logger.info("[SOC_LOT] Parsing structured LOT results for '%s'...", country)
    parsed = parse_overlay_output(overlay_output, drugs)

    rows: list[LotRow] = []
    for drug in drugs:
        info = parsed.get(drug, {"lot_type": "", "rationale": "", "confidence": None})
        lot_score = map_lot_type_to_score(info["lot_type"])
        rows.append(
            LotRow(
                drug_name=drug,
                country=country,
                lot_score=lot_score,
                lot_type=info["lot_type"],
                rationale=info["rationale"],
                confidence=info["confidence"],
                final_lot_score=None,  # filled in once, after all countries are processed
            )
        )
    return rows


# ==============================
# OUTPUT: BIGQUERY
# ==============================
LOT_RESULTS_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("drug_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("country", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("lot_score", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("lot_type", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("rationale", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("confidence", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("final_lot_score", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
]


def push_results_to_bigquery(rows: list[LotRow]) -> None:
    """Pushes all LOT result rows to the configured BigQuery table.

    Reuses the ``gcp_utils.get_bq_client`` credential pattern and writes in
    append mode, creating the table if it does not already exist, matching
    ``gcp_utils.append_dimension_score_to_bigquery``.
    """
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{LOT_TABLE}"
    timestamp = datetime.now(timezone.utc).isoformat()

    payload = [
        {
            "drug_name": r.drug_name,
            "country": r.country,
            "lot_score": r.lot_score,
            "lot_type": r.lot_type,
            "rationale": r.rationale,
            "confidence": r.confidence,
            "final_lot_score": r.final_lot_score,
            "timestamp": timestamp,
        }
        for r in rows
    ]

    bq_client = get_bq_client()
    job_config = bigquery.LoadJobConfig(
        schema=LOT_RESULTS_SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
    )
    load_job = bq_client.load_table_from_json(payload, table_id, job_config=job_config)
    load_job.result()
    logger.info("[SOC_LOT] Pushed %d row(s) to %s", len(payload), table_id)
