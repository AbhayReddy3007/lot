"""Prompt-based rationale generator for serious safety profile scoring."""

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

    Returns a small dictionary suitable for embedding in the rationale prompt.
    """
    if not isinstance(data, dict):
        return {}

    molecule = data.get("molecule_name")
    safety_score = data.get("safety_score")
    sae_agg = data.get("sae_aggregation")

    # Condense SAE event categorization to top material events
    raw_events = data.get("sae_event_categorization")
    events = []
    for ev in (raw_events or [])[:25]:
        try:
            events.append({
                "sae_name": ev.get("sae_name"),
                "severity_category": ev.get("severity_category"),
                "expectedness": ev.get("expectedness_classification"),
                "number_of_studies": ev.get("number_of_studies"),
            })
        except Exception:
            continue

    regulatory = data.get("regulatory_impact") or {}
    post_marketing = data.get("post_marketing_safety") or {}

    # Prepare top trials: select key fields, sort by normalized_weight desc, and take top 10
    raw_trials = sae_agg.get("trials") or []
    trials = []
    for tr in (raw_trials or []):
        try:
            trials.append({
                "trial_id": tr.get("Trial ID") or tr.get("trial_id") or tr.get("NCT_ID") or None,
                "phase": tr.get("Phase"),
                "size": tr.get("Size"),
                "primary_region": tr.get("Primary Region") or tr.get("primary_region"),
                "dosage": tr.get("Dosage"),
                "sae_rate_drug": tr.get("SAE Rate Drug (%)"),
                "sae_rate_control": tr.get("SAE Rate Control (%)"),
                "delta_sae": tr.get("Delta SAE (%)"),
                "normalized_weight": tr.get("normalized_weight") or tr.get("normalizedWeight") or 0,
            })
        except Exception:
            continue

    try:
        sorted_trials = sorted(trials, key=lambda x: float(x.get("normalized_weight") or 0), reverse=True)
        top_trials = sorted_trials[:10]
    except Exception:
        top_trials = trials[:10]

    return {
        "molecule_name": molecule,
        "safety_score": {
            "score": safety_score.get("score"),
            "key_drivers": safety_score.get("key_drivers", []),
            "adjustments_applied": safety_score.get("adjustments_applied",[]),
            "reasoning": safety_score.get("reasoning",{})
        },
        "num_trials": len(raw_trials),
        "top_trials": top_trials,
        "sae_aggregation": {
            "sae_rate": sae_agg.get("sae_rate"),
            "control_rate": sae_agg.get("control_rate"),
            "delta_rate": sae_agg.get("delta_rate"),
        },
        "top_events": events,
        "regulatory_consequence": regulatory.get("regulatory_consequence"),
        "post_marketing": {
            "evaluated": post_marketing.get("evaluated"),
            "new_serious_risks": post_marketing.get("new_serious_risks"),
            "risk_level": post_marketing.get("risk_level"),
        },
    }


async def generate_serious_safety_rationale(
    data: dict[str, Any],
    model_name: str = GEMINI_FLASH_PREVIEW_MODEL,
 ) -> tuple[str, dict[str, Any]]:
    """Generate a concise qualitative rationale for serious safety score."""
    # Reduce the input to only the fields relevant for a concise rationale.
    payload = _prepare_prompt_payload(data)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return NOT_GENERATED_MESSAGE, payload

    prompt = f"""
You are a concise medical insights writer. Produce a short, plain-text rationale (one sentence or a very short paragraph) that explains the main driving forces behind the Serious Safety Profile score for a drug.

Input (JSON):
{json.dumps(payload, indent=2)}

Requirements:
- Explain only the primary drivers that push the score higher or lower (e.g., consistent positive delta SAE, repeated life‑threatening events, regulatory consequence, or strong post‑marketing signals).
- Focus on the safety_information provided with adjustments applied, reasoning, and key drivers to get the better idea on scoring
- Mention uncertainty when trial data are limited or rates are missing.
- Avoid methodology, scoring framework names, internal field names, or tables.
- Do not mention or cite any specific trial or event
- Use plain language appropriate for a Medical Affairs audience.
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
            config = config,
        )
        rationale = (response.text or "").strip()
        if rationale:
            return rationale, payload
        logger.warning("[SERIOUS_SAFETY][RATIONALE] Prompt output was empty; rationale was not generated.")
    except Exception:
        logger.exception("[SERIOUS_SAFETY][RATIONALE] Prompt generation failed; rationale was not generated.")

    return NOT_GENERATED_MESSAGE, payload
