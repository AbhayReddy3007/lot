"""
generate_lot_report.py
───────────────────────
Reads scored data from BigQuery `LOT_TABLE` (populated by
``line_of_treatment.py`` / ``lot_scoring.py``) and generates one professional
PDF report **per drug** using Gemini for narrative generation.

For each drug, the LATEST row per (drug_name, country) — by timestamp — is
used, together with the most recently computed final_lot_score for that
drug.

The report pulls ALL available fields from the LOT_TABLE (per-country
lot_score, lot_type, rationale, confidence, and the aggregate
final_lot_score) and asks Gemini to turn them into a business-facing
narrative.

Report structure (single-drug, business-facing):
  - Title block (drug name + final LOT score)
  - Key Line-of-Treatment Findings
  - Country-by-Country LOT Breakdown (table + rationale)
  - Insights and Implications
  - Line-of-Treatment Profile Summary
  - LOT scoring reference table (end of document)

Each report is written to:
    {LOT_REPORT_PATH}/{drug_name}/Line_of_Treatment.pdf

Usage:
    # All drugs currently in LOT_TABLE — one PDF each
    python generate_lot_report.py

    # One specific drug
    python generate_lot_report.py --drug Semaglutide

    # Two drugs — two separate PDFs
    python generate_lot_report.py --drug "Semaglutide,Tirzepatide"
"""

import os
import re
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google.cloud import bigquery

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable,
)

from google import genai as genai_client
from google.genai import types

from medical_potential.config import (
    BQ_DATASET_ID,
    GEMINI_FLASH_PREVIEW_MODEL,
    LOT_REPORT_PATH,
    LOT_TABLE,
    PROJECT_ID,
)
from medical_potential.gcp_utils import get_bq_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = GEMINI_FLASH_PREVIEW_MODEL

REPORT_FILE_NAME = "Line_of_Treatment.pdf"

# Minimum character count to consider a rationale field "sufficient"
MIN_RATIONALE_LENGTH = 40

# ── Colors ────────────────────────────────────────────────────────────────────
DARK_BLUE = colors.HexColor("#1F3864")
LIGHT_BLUE_BG = colors.HexColor("#E8EDF3")
ACCENT_BLUE = colors.HexColor("#2E5FA3")
WHITE = colors.white
LIGHT_GRAY = colors.HexColor("#666666")
DIVIDER_COLOR = colors.HexColor("#D0D7E3")

LOT_SCORE_COLORS = {
    5: colors.HexColor("#008000"),
    4: colors.HexColor("#4CAF50"),
    3: colors.HexColor("#CC9900"),
    2: colors.HexColor("#E65100"),
    1: colors.HexColor("#CC0000"),
}

LOT_SCORE_LABEL = {
    5: "First-line standard of care",
    4: "Strong first-line alternative / dominant second-line",
    3: "Second-line option",
    2: "Third-line or restricted niche use",
    1: "Salvage / last-resort use",
}


# ── Gemini helpers ────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    """Call Gemini for narrative generation."""
    client = genai_client.Client(api_key=API_KEY)
    config = types.GenerateContentConfig(temperature=0.3)
    response = client.models.generate_content(model=MODEL, contents=prompt, config=config)
    return response.text.strip() if response.text else ""


def _extract_json(text: str):
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_from_bigquery(drugs: list[str] | None = None) -> list[dict]:
    """
    Load the LATEST row per (drug_name, country) from LOT_TABLE.
    If `drugs` is provided, only those drugs are fetched.
    """
    bq_client = get_bq_client()
    table_ref = f"`{PROJECT_ID}.{BQ_DATASET_ID}.{LOT_TABLE}`"

    drug_filter = ""
    query_params = []
    if drugs:
        placeholders = ", ".join(f"@drug_{i}" for i in range(len(drugs)))
        drug_filter = f"AND drug_name IN ({placeholders})"
        query_params = [
            bigquery.ScalarQueryParameter(f"drug_{i}", "STRING", d.strip())
            for i, d in enumerate(drugs)
        ]

    query = f"""
        WITH ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY drug_name, country ORDER BY timestamp DESC
                ) AS _rn
            FROM {table_ref}
            WHERE drug_name IS NOT NULL {drug_filter}
        )
        SELECT * EXCEPT(_rn) FROM ranked WHERE _rn = 1
        ORDER BY drug_name, country
    """
    job_config = bigquery.QueryJobConfig(query_parameters=query_params) if query_params else None

    logger.info("[LOT_REPORT] Loading latest LOT data from %s...", LOT_TABLE)
    rows = [dict(row) for row in bq_client.query(query, job_config=job_config).result()]
    logger.info("[LOT_REPORT] Loaded %d row(s) across all drug/country combinations.", len(rows))
    return rows


def group_rows_by_drug(rows: list[dict]) -> dict[str, dict]:
    """Groups flat BQ rows into {drug_name: {"countries": [...], "final_lot_score": ...}}."""
    grouped: dict[str, dict] = {}
    for row in rows:
        drug = row.get("drug_name")
        if not drug:
            continue
        bucket = grouped.setdefault(drug, {"countries": [], "final_lot_score": None, "_latest_ts": None})
        bucket["countries"].append(row)

        ts = row.get("timestamp")
        if ts is not None and (bucket["_latest_ts"] is None or ts > bucket["_latest_ts"]):
            bucket["_latest_ts"] = ts
            if row.get("final_lot_score") is not None:
                bucket["final_lot_score"] = row.get("final_lot_score")

    for bucket in grouped.values():
        bucket["countries"].sort(key=lambda r: r.get("country") or "")
        bucket.pop("_latest_ts", None)

    return grouped


# ── Single-drug statistics ────────────────────────────────────────────────────

def extract_drug_stats(drug_name: str, bucket: dict) -> dict:
    """Extract ALL relevant fields for a single drug across all its countries."""
    def safe(val, fallback="N/A"):
        return str(val).strip() if val is not None and str(val).strip() else fallback

    countries = []
    for row in bucket["countries"]:
        score_raw = row.get("lot_score")
        score_int = None
        try:
            score_int = int(float(score_raw))
        except (ValueError, TypeError):
            pass

        confidence_raw = row.get("confidence")
        confidence_int = None
        try:
            confidence_int = int(float(confidence_raw))
        except (ValueError, TypeError):
            pass

        countries.append({
            "country": safe(row.get("country")),
            "lot_score": score_int,
            "lot_score_label": LOT_SCORE_LABEL.get(score_int, "N/A") if score_int else "N/A",
            "lot_type": safe(row.get("lot_type")),
            "rationale": safe(row.get("rationale"), ""),
            "confidence": confidence_int,
        })

    final_score_raw = bucket.get("final_lot_score")
    final_score = None
    try:
        final_score = round(float(final_score_raw), 2)
    except (ValueError, TypeError):
        pass

    return {
        "drug_name": drug_name,
        "countries": countries,
        "final_lot_score": final_score,
    }


def _is_data_sufficient(stats: dict) -> bool:
    """Check whether enough countries have a usable score + rationale."""
    usable = [
        c for c in stats["countries"]
        if c["lot_score"] is not None and len(c["rationale"]) >= MIN_RATIONALE_LENGTH
    ]
    return len(usable) > 0


# ── LLM narrative (single drug) ───────────────────────────────────────────────

def generate_lot_narrative(stats: dict) -> dict:
    """
    Generate a business-focused Line-of-Treatment report narrative for
    Market Access / Medical Affairs.

    Returns a dict with: key_findings, insights_implications, profile_summary.
    """
    data_parts = [f"Drug: {stats['drug_name']}"]
    if stats["final_lot_score"] is not None:
        data_parts.append(f"Aggregate Final LOT Score: {stats['final_lot_score']} (out of 5)")

    country_lines = []
    for c in stats["countries"]:
        line = f"- {c['country']}: LOT Score {c['lot_score']}/5 ({c['lot_type'] or 'N/A'}), Confidence {c['confidence']}%"
        if c["rationale"]:
            line += f" — Rationale: {c['rationale']}"
        country_lines.append(line)

    prompt = f"""You are a business-focused market access analyst.

Goal: Create a concise, business-facing Line of Treatment (LOT) report for {stats['drug_name']}
highlighting key findings and insights derived from the provided per-country LOT classification
data. The report is intended for a Market Access / Medical Affairs business audience.

Context: The data comes from a structured pipeline that classifies where {stats['drug_name']}
sits in each country's treatment pathway (e.g. first-line, second-line, third-line, salvage),
based on that country's Standard of Care guidance and the drug's mechanism of action. The
audience is non-technical and not familiar with internal scoring methodology.

Source: Use only the provided data as the source of truth. Do not introduce external
assumptions unless clearly derived from the data.

=== PROVIDED DATA ===
{chr(10).join(data_parts)}

Per-country breakdown:
{chr(10).join(country_lines) if country_lines else 'Not available'}

=== INSTRUCTIONS ===

1. Start with a section: "Key Line of Treatment Findings for {stats['drug_name']}"
   - Summarize the most important observations about where this drug sits in treatment
     pathways across the countries analysed
   - Include the aggregate final LOT score (e.g., "scored X out of 5") in simple terms
     early in the findings — but do NOT explain how the score was derived or its weighting
   - Highlight:
     - Where the drug is positioned (first-line vs later-line) in the US versus other markets
     - Any notable geographic variation in treatment-line placement
     - Overall confidence in the classifications
   - Focus on what the data shows, not how it was calculated

2. Follow with: "Insights and Implications"
   - Translate LOT findings into business-relevant insights
   - Highlight:
     - What the treatment-line positioning means for market access and reimbursement
     - Key differences between geographies and why they might matter commercially
     - Strategic implications for positioning, launch sequencing, or payer engagement
   - Keep insights simple, clear, and actionable

3. Include: "Line of Treatment Profile Summary"
   - Provide a high-level summary of the drug's overall treatment-line positioning
   - Comment on consistency of positioning across countries if available
   - Briefly mention the aggregate final LOT score in simple terms without explaining
     scoring methodology

Language and Style:
- Do NOT use internal jargon, scoring framework names, or technical modeling terms
  (e.g. do not mention "US_WEIGHT", "OTHER_COUNTRY_WEIGHT", or how the aggregate is computed)
- Use clear, simple, business-friendly language
- Translate classification findings into plain-language market access impact

Tone: Professional, objective, and insight-driven. Focus on clarity, relevance, and
business impact.

Respond ONLY with a valid JSON object (no markdown fences, no extra text):
{{
  "key_findings": {{
    "summary_bullets": [
      "Key finding 1 — MUST state the aggregate final LOT score as X out of 5 (use the exact score from the data) and what it indicates in plain terms",
      "Key finding 2 about US treatment-line positioning specifically",
      "Key finding 3 about positioning in other markets and any notable variation",
      "Key finding 4 about overall confidence in the classifications",
      "Key finding 5 about any other important observation"
    ],
    "geographic_variation_detail": "2-3 sentences in plain language about how treatment-line placement varies (or doesn't) across the countries analysed",
    "therapy_line_detail": "2-3 sentences describing where the drug typically sits in the treatment pathway and what that means for patients reaching it"
  }},
  "insights_implications": {{
    "market_access_impact": "2-3 sentences on what this treatment-line positioning means for reimbursement and market access",
    "geographic_strategy": "2-3 sentences on how the geographic differences could shape launch sequencing or country-specific strategy",
    "positioning_implications": "2-3 sentences on how this affects competitive positioning versus other therapies in the same treatment lines",
    "strategic_recommendation": "2-3 sentences with a business-relevant recommendation given this LOT profile"
  }},
  "profile_summary": {{
    "overall_assessment": "3-4 sentences providing a high-level summary of the drug's overall treatment-line positioning, written for a business audience",
    "cross_country_consistency": "1-2 sentences commenting on whether treatment-line positioning is consistent across countries",
    "score_context": "1-2 sentences stating the aggregate final LOT score explicitly (e.g., 'The drug scored 3.2 out of 5 on the Line of Treatment scale, indicating...'). ALWAYS include the numeric score (X out of 5). Do NOT explain how the score was derived."
  }}
}}"""

    text = call_gemini(prompt)
    result = _extract_json(text)
    if result:
        return result

    # Fallback if Gemini fails or returns unparseable output
    return {
        "key_findings": {
            "summary_bullets": [
                f"{stats['drug_name']} received an aggregate final LOT score of {stats['final_lot_score']} out of 5.",
                f"Countries analysed: {', '.join(c['country'] for c in stats['countries']) or 'N/A'}.",
            ],
            "geographic_variation_detail": "See country-by-country breakdown below.",
            "therapy_line_detail": "See country-by-country breakdown below.",
        },
        "insights_implications": {
            "market_access_impact": "See detailed analysis.",
            "geographic_strategy": "See detailed analysis.",
            "positioning_implications": "See detailed analysis.",
            "strategic_recommendation": "See detailed analysis.",
        },
        "profile_summary": {
            "overall_assessment": f"{stats['drug_name']} received an aggregate final LOT score of {stats['final_lot_score']} out of 5.",
            "cross_country_consistency": "Consistency is based on available country-level classifications.",
            "score_context": f"The drug scored {stats['final_lot_score']} out of 5 on the Line of Treatment scale.",
        },
    }


# ── Style helpers ─────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Normal"],
            fontSize=18, leading=24, textColor=DARK_BLUE,
            alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"],
            fontSize=9, leading=12, textColor=LIGHT_GRAY,
            alignment=TA_CENTER, fontName="Helvetica", spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Normal"],
            fontSize=13, leading=16, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=10, leading=14, textColor=colors.HexColor("#333333"),
            fontName="Helvetica", spaceAfter=4, alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"],
            fontSize=9, leading=13, textColor=colors.HexColor("#333333"),
            fontName="Helvetica", spaceAfter=3, leftIndent=18,
            bulletIndent=6,
        ),
        "footer": ParagraphStyle(
            "Footer", parent=base["Normal"],
            fontSize=7, leading=10, textColor=colors.HexColor("#999999"),
            fontName="Helvetica", alignment=TA_CENTER, spaceBefore=10,
        ),
        "cell": ParagraphStyle(
            "Cell", parent=base["Normal"],
            fontSize=8, leading=11, textColor=colors.HexColor("#333333"),
            fontName="Helvetica", alignment=TA_CENTER,
        ),
        "cell_header": ParagraphStyle(
            "CellHeader", parent=base["Normal"],
            fontSize=8, leading=11, textColor=WHITE,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "section_label": ParagraphStyle(
            "SectionLabel", parent=base["Normal"],
            fontSize=10, leading=13, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", spaceAfter=2, leftIndent=6,
        ),
    }


def _render_bullets(items: list, styles: dict, story: list):
    for item in items:
        if item and str(item).strip():
            story.append(Paragraph(f"&#8226; {item}", styles["bullet"]))


def _country_breakdown_table(countries: list[dict], styles: dict) -> Table:
    header = [
        Paragraph("Country", styles["cell_header"]),
        Paragraph("LOT Score", styles["cell_header"]),
        Paragraph("LOT Type", styles["cell_header"]),
        Paragraph("Confidence", styles["cell_header"]),
    ]
    rows = [header]
    for c in countries:
        score_display = f"{c['lot_score']}/5" if c["lot_score"] is not None else "N/A"
        confidence_display = f"{c['confidence']}%" if c["confidence"] is not None else "N/A"
        rows.append([
            Paragraph(c["country"], ParagraphStyle("CBCountry", parent=styles["cell"], alignment=TA_LEFT)),
            Paragraph(score_display, styles["cell"]),
            Paragraph(c["lot_type"] or "N/A", ParagraphStyle("CBType", parent=styles["cell"], alignment=TA_LEFT)),
            Paragraph(confidence_display, styles["cell"]),
        ])

    tbl = Table(rows, colWidths=[1.4 * inch, 0.9 * inch, 3.0 * inch, 1.1 * inch])
    row_bgs = [
        ("BACKGROUND", (0, i), (-1, i), LIGHT_BLUE_BG if i % 2 == 0 else WHITE)
        for i in range(1, len(rows))
    ]
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        *row_bgs,
    ]))
    return tbl


def _scoring_framework_table(styles: dict) -> Table:
    """Render the LOT scoring reference table."""
    framework = [
        ("5", LOT_SCORE_LABEL[5]),
        ("4", LOT_SCORE_LABEL[4]),
        ("3", LOT_SCORE_LABEL[3]),
        ("2", LOT_SCORE_LABEL[2]),
        ("1", LOT_SCORE_LABEL[1]),
    ]
    header = [
        Paragraph("Score", styles["cell_header"]),
        Paragraph("Classification", styles["cell_header"]),
    ]
    rows = [header]
    for sc, label in framework:
        rows.append([
            Paragraph(sc, ParagraphStyle("FWScore", parent=styles["cell"], fontName="Helvetica")),
            Paragraph(label, ParagraphStyle("FWLabel", parent=styles["cell"], alignment=TA_LEFT)),
        ])

    tbl = Table(rows, colWidths=[0.7 * inch, 5.9 * inch])
    row_bgs = [
        ("BACKGROUND", (0, i), (-1, i), LIGHT_BLUE_BG if i % 2 == 0 else WHITE)
        for i in range(1, len(rows))
    ]
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        *row_bgs,
    ]))
    return tbl


# ── Single-drug report builder ────────────────────────────────────────────────

def build_single_drug_report(stats: dict, narrative: dict, output_path: str):
    """Build and save a business-focused PDF report for one drug."""
    styles = build_styles()
    story = []

    report_title = f"{stats['drug_name']} Line of Treatment Report"

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        title=report_title,
        author="LOT Scorer",
    )

    # ── Title block ───────────────────────────────────────────────────────
    story.append(Paragraph(report_title, styles["title"]))
    final_score_display = stats["final_lot_score"] if stats["final_lot_score"] is not None else "N/A"
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')}  •  Final LOT Score: {final_score_display} / 5",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=DARK_BLUE, spaceAfter=12))

    # ══════════════════════════════════════════════════════════════════
    # Key Line of Treatment Findings
    # ══════════════════════════════════════════════════════════════════
    story.append(Paragraph(f"Key Line of Treatment Findings for {stats['drug_name']}", styles["h2"]))

    key_findings = narrative.get("key_findings", {})
    if isinstance(key_findings, dict):
        bullets = key_findings.get("summary_bullets", [])
        if isinstance(bullets, list):
            _render_bullets(bullets, styles, story)
        story.append(Spacer(1, 6))

        geo_detail = key_findings.get("geographic_variation_detail", "")
        if geo_detail:
            story.append(Paragraph("<b>Geographic Variation:</b>", styles["section_label"]))
            story.append(Paragraph(geo_detail, styles["body"]))
            story.append(Spacer(1, 4))

        therapy_detail = key_findings.get("therapy_line_detail", "")
        if therapy_detail:
            story.append(Paragraph("<b>Treatment-Line Positioning:</b>", styles["section_label"]))
            story.append(Paragraph(therapy_detail, styles["body"]))
    elif isinstance(key_findings, str):
        story.append(Paragraph(key_findings, styles["body"]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════
    # Country-by-Country LOT Breakdown
    # ══════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Country-by-Country LOT Breakdown", styles["h2"]))
    if stats["countries"]:
        story.append(_country_breakdown_table(stats["countries"], styles))
        story.append(Spacer(1, 8))

        for c in stats["countries"]:
            if c["rationale"]:
                story.append(Paragraph(f"<b>{c['country']}:</b> {c['rationale']}", styles["body"]))
                story.append(Spacer(1, 3))
    else:
        story.append(Paragraph("No country-level data available.", styles["body"]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════
    # Insights and Implications
    # ══════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Insights and Implications", styles["h2"]))

    insights = narrative.get("insights_implications", {})
    if isinstance(insights, dict):
        market_access = insights.get("market_access_impact", "")
        if market_access:
            story.append(Paragraph(f"<b>Market Access Impact:</b> {market_access}", styles["body"]))
            story.append(Spacer(1, 4))

        geo_strategy = insights.get("geographic_strategy", "")
        if geo_strategy:
            story.append(Paragraph(f"<b>Geographic Strategy:</b> {geo_strategy}", styles["body"]))
            story.append(Spacer(1, 4))

        positioning = insights.get("positioning_implications", "")
        if positioning:
            story.append(Paragraph(f"<b>Competitive Positioning:</b> {positioning}", styles["body"]))
            story.append(Spacer(1, 4))

        recommendation = insights.get("strategic_recommendation", "")
        if recommendation:
            story.append(Paragraph(f"<b>Strategic Recommendation:</b> {recommendation}", styles["body"]))
    elif isinstance(insights, str):
        story.append(Paragraph(insights, styles["body"]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════
    # Line of Treatment Profile Summary
    # ══════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Line of Treatment Profile Summary", styles["h2"]))

    profile_summary = narrative.get("profile_summary", {})
    if isinstance(profile_summary, dict):
        overall = profile_summary.get("overall_assessment", "")
        if overall:
            story.append(Paragraph(overall, styles["body"]))
            story.append(Spacer(1, 4))

        consistency = profile_summary.get("cross_country_consistency", "")
        if consistency:
            story.append(Paragraph(consistency, styles["body"]))
            story.append(Spacer(1, 4))

        score_ctx = profile_summary.get("score_context", "")
        if score_ctx:
            story.append(Paragraph(score_ctx, styles["body"]))
    elif isinstance(profile_summary, str):
        story.append(Paragraph(profile_summary, styles["body"]))
    story.append(Spacer(1, 8))

    # ── Scoring reference table ─────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Line of Treatment Scoring Reference", styles["h2"]))
    story.append(_scoring_framework_table(styles))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor("#CCCCCC"), spaceBefore=14,
    ))
    story.append(Paragraph(
        "This report was auto-generated from Line of Treatment analysis output. "
        "For internal use only.",
        styles["footer"],
    ))

    doc.build(story)
    logger.info("[LOT_REPORT] Report saved -> %s", output_path)


# ── Public entry point ─────────────────────────────────────────────────────────

def generate_lot_reports(drugs: list[str] | None = None) -> list[str]:
    """
    Generate one PDF report per drug, saved to
    {LOT_REPORT_PATH}/{drug_name}/Line_of_Treatment.pdf.

    Args:
        drugs: List of drug names to report on. None = all drugs in LOT_TABLE.

    Returns:
        List of output PDF paths that were successfully created.
    """
    if not API_KEY:
        logger.warning("[LOT_REPORT] GEMINI_API_KEY not set — skipping report generation.")
        return []

    rows = load_from_bigquery(drugs)
    if not rows:
        logger.warning("[LOT_REPORT] No data found in %s — skipping.", LOT_TABLE)
        return []

    grouped = group_rows_by_drug(rows)

    output_paths = []
    for drug_name, bucket in grouped.items():
        logger.info("[LOT_REPORT] Processing: %s", drug_name)

        stats = extract_drug_stats(drug_name, bucket)
        if not _is_data_sufficient(stats):
            logger.warning("[LOT_REPORT] Insufficient LOT data for '%s' — generating report from what's available.", drug_name)

        logger.info("[LOT_REPORT] Generating narrative with Gemini for '%s'...", drug_name)
        narrative = generate_lot_narrative(stats)

        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
        out_dir = Path(LOT_REPORT_PATH) / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / REPORT_FILE_NAME)

        build_single_drug_report(stats, narrative, output_path)
        output_paths.append(output_path)

    logger.info("[LOT_REPORT] Done. %d report(s) generated.", len(output_paths))
    return output_paths


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate one Line of Treatment PDF per drug from BigQuery data."
    )
    parser.add_argument(
        "--drug", "-d",
        default=None,
        help=(
            "Comma-separated drug name(s) to report on. "
            "E.g. --drug Semaglutide  or  --drug 'Semaglutide,Tirzepatide'. "
            "Omit to process all drugs in the LOT_TABLE."
        ),
    )
    args = parser.parse_args()

    drugs = None
    if args.drug:
        drugs = [d.strip() for d in args.drug.split(",") if d.strip()]

    generate_lot_reports(drugs=drugs)


if __name__ == "__main__":
    main()
