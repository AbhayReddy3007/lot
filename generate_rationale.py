"""Prompt-based rationale generator for the final Line of Treatment (LOT) score.

Mirrors the style of the serious-safety-profile rationale generator: reduce
the scored data down to the minimal fields needed, hand them to Gemini, and
return a short plain-text rationale plus the payload that was sent.

Place this module alongside ``lot_scoring.py`` and ``generate_lot_report.py``
at ``medical_potential/line_of_treatment/generate_rationale.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import json
from typing import Any

from google import genai
from google.genai import types

from medical_potential.config import GEMINI_FLASH_PREVIEW_MODEL

logger = logging.getLogger(__name__)

NOT_GENERATED_MESSAGE = "Rationale has not been generated."


def _prepare_prompt_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Select and reduce input data to the minimal fields needed for a concise rationale.

    ``data`` is expected to look like the ``stats`` dict built by
    ``generate_lot_report.extract_drug_stats``:
        {
            "drug_name": str,
            "final_lot_score": float | None,
            "countries": [
                {"country": str, "lot_score": int | None, "lot_type": str,
                 "confidence": int | None, "rationale": str},
                ...
            ],
        }

    Returns a small dictionary suitable for embedding in the rationale prompt.
    """
    if not isinstance(data, dict):
        return {}

    drug_name = data.get("drug_name")
    final_lot_score = data.get("final_lot_score")

    # Condense per-country breakdown to the fields that actually matter for
    # explaining the aggregate score, capped so the prompt stays small.
    raw_countries = data.get("countries") or []
    countries = []
    for c in raw_countries[:25]:
        try:
            rationale = (c.get("rationale") or "").strip()
            countries.append({
                "country": c.get("country"),
                "lot_score": c.get("lot_score"),
                "lot_type": c.get("lot_type"),
                "confidence": c.get("confidence"),
                # Keep only a short excerpt — the full rationale is already
                # shown elsewhere in the report and isn't needed here.
                "rationale_excerpt": rationale[:200],
            })
        except Exception:
            continue

    try:
        sorted_countries = sorted(
            countries,
            key=lambda x: (x.get("confidence") if x.get("confidence") is not None else -1),
            reverse=True,
        )
        top_countries = sorted_countries[:10]
    except Exception:
        top_countries = countries[:10]

    scored = [c for c in countries if c.get("lot_score") is not None]
    missing_count = len(countries) - len(scored)

    return {
        "molecule_name": drug_name,
        "final_lot_score": final_lot_score,
        "num_countries": len(raw_countries),
        "num_countries_scored": len(scored),
        "num_countries_missing_score": missing_count,
        "top_countries": top_countries,
    }


async def generate_final_lot_rationale(
    data: dict[str, Any],
    model_name: str = GEMINI_FLASH_PREVIEW_MODEL,
) -> tuple[str, dict[str, Any]]:
    """Generate a concise qualitative rationale for a drug's final LOT score."""
    # Reduce the input to only the fields relevant for a concise rationale.
    payload = _prepare_prompt_payload(data)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return NOT_GENERATED_MESSAGE, payload

    prompt = f"""
You are a concise medical insights writer. Produce a short, plain-text rationale (one sentence or a very short paragraph) that explains the main driving forces behind the Final Line of Treatment (LOT) score for a drug.

Input (JSON):
{json.dumps(payload, indent=2)}

Requirements:
- Explain only the primary drivers behind the final LOT score (e.g., where the drug sits earliest/latest in the treatment pathway, which countries pulled the score up or down, and overall confidence in those classifications).
- Focus on what the per-country data shows, not on how the aggregate score was calculated or weighted.
- Mention uncertainty when country coverage is limited or confidence is low.
- Avoid methodology, scoring framework names, internal field names, or tables.
- Do not mention or cite any specific country-level rationale text verbatim.
- Use plain language appropriate for a Medical Affairs / Market Access audience.
- Return plain text only.

Note:
Strictly limit to 50 words, anything longer will be penalised
""".strip()

    client = genai.Client(api_key=api_key)
    try:
        config = types.GenerateContentConfig(temperature=0)
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model_name,
            contents=contents,
            config=config,
        )
        rationale = (response.text or "").strip()
        if rationale:
            return rationale, payload
        logger.warning("[FINAL_LOT][RATIONALE] Prompt output was empty; rationale was not generated.")
    except Exception:
        logger.exception("[FINAL_LOT][RATIONALE] Prompt generation failed; rationale was not generated.")

    return NOT_GENERATED_MESSAGE, payload
