"""Entry point for the SoC → Lines-of-Therapy (LOT) scoring pipeline.

All the actual logic (config, GCS/BigQuery/Gemini calls, benchmark caching,
scoring, and the BigQuery push) lives in ``lot_scoring.py``. PDF report
generation lives in ``generate_lot_report.py``. This module just wires those
pieces together and exposes ``main()`` as the CLI entry point.

The drug to score is read from ``LOT_DRUG`` in config — this pipeline scores
exactly one drug per run.

Every run scores the drug and updates BigQuery, then generates the Line of
Treatment PDF report by default (via ``generate_lot_report.py``), using the
values that were just written to BigQuery. Pass ``--no-report`` to skip
report generation and only run scoring + the BigQuery push.

Place this module at ``medical_potential/line_of_treatment/line_of_treatment.py``
alongside ``medical_potential/line_of_treatment/lot_scoring.py`` and
``medical_potential/line_of_treatment/generate_lot_report.py``.
"""

from __future__ import annotations

import argparse
import logging

from medical_potential.config import GCS_BUCKET, GCS_SOC_BASE_PATH, LOT_DRUG
from medical_potential.line_of_treatment.generate_lot_report import generate_lot_reports
from medical_potential.line_of_treatment.lot_scoring import (
    LotRow,
    build_overlay_prompt,
    compute_final_lot_scores_per_drug,
    discover_countries_gcs,
    lookup_moa,
    process_country,
    push_results_to_bigquery,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_scoring() -> None:
    """Scores LOT_DRUG across all countries and pushes the results to BigQuery."""
    if not LOT_DRUG:
        raise SystemExit("LOT_DRUG is not set in config.py — cannot run scoring.")

    drugs = [LOT_DRUG]
    logger.info("[SOC_LOT] Drug to analyse: %s", LOT_DRUG)

    logger.info("[SOC_LOT] Looking up MoA from BigQuery...")
    drug_moa = lookup_moa(drugs)
    for drug, info in drug_moa.items():
        detail = f" ({info['moa_detailed']})" if info["moa_detailed"] else ""
        logger.info("[SOC_LOT] %s -> %s%s", drug, info["moa"], detail)

    overlay_prompt = build_overlay_prompt(drugs, drug_moa)

    logger.info("[SOC_LOT] Scanning gs://%s/%s...", GCS_BUCKET, GCS_SOC_BASE_PATH)
    countries = discover_countries_gcs()
    logger.info("[SOC_LOT] Found %d country folder(s): %s", len(countries), ", ".join(countries.keys()))

    all_rows: list[LotRow] = []
    for country, blob_names in countries.items():
        try:
            all_rows.extend(process_country(country, blob_names, drugs, overlay_prompt))
        except Exception:
            logger.exception("[SOC_LOT] Failed processing country '%s'", country)

    # final_lot_score is a per-drug aggregate across all countries:
    # (US lot_score x US_WEIGHT) + OTHER_COUNTRY_WEIGHT x sum(lot_score for every other country)
    final_scores = compute_final_lot_scores_per_drug(all_rows)
    for r in all_rows:
        r.final_lot_score = final_scores.get(r.drug_name)

    logger.info("[SOC_LOT] Pushing results to BigQuery...")
    push_results_to_bigquery(all_rows)

    logger.info("[SOC_LOT] SUMMARY")
    for country in countries:
        n = sum(1 for r in all_rows if r.country == country)
        logger.info("[SOC_LOT]   %s: %d row(s)", country, n)


def run_report() -> None:
    """Generates the Line of Treatment PDF for LOT_DRUG from the latest BigQuery data."""
    if not LOT_DRUG:
        raise SystemExit("LOT_DRUG is not set in config.py — cannot generate a report.")

    logger.info("[LOT_REPORT] Generating report for: %s", LOT_DRUG)
    generate_lot_reports([LOT_DRUG])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score LOT_DRUG's Line of Treatment across countries, push to BigQuery, and generate the PDF report."
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip PDF report generation and only run scoring + the BigQuery push.",
    )
    args = parser.parse_args()

    run_scoring()

    if not args.no_report:
        run_report()


if __name__ == "__main__":
    main()
