
"""
trial_fetcher.py – Unified clinical trial fetcher.

Queries ClinicalTrials.gov (CTGOV), EU CTIS, EudraCT, and CTRI for a given
molecule name, then writes a JSON file with one dict per trial and
the following standardised columns:

    molecule_name, registry_source, trial_id, acronym, dosage, phase,
    trial_title, trial_study, trial_size, trial_location, trial_start_date,
    trial_completion_date, phase_status,
    hba1c_change_pct, hba1c_duration, hba1c_rationale, hba1c_confidence,
    weight_change_pct, weight_duration, weight_rationale, weight_confidence,
    alt_reduction_pct, alt_duration, alt_rationale, alt_confidence,
    mash_change_pct, mash_duration, mash_rationale, mash_confidence,
    company_name, source_url

Usage:
    python trial_fetcher.py <molecule_name> [--max-records N] [--out output.json]
    python trial_fetcher.py semaglutide --max-records 50 --out semaglutide_trials.json
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional

import requests
import asyncio as _asyncio

# ── optional import of local registry modules ─────────────────────────────────
try:
    from . import ctgov_trials as _ctgov
except ImportError:
    _ctgov = None  # type: ignore

try:
    from . import ctis_drug_trials as _ctis
except ImportError:
    _ctis = None  # type: ignore

try:
    from .nice_trials import scan_nice_appraisals as _nice_scan
    _HAS_NICE = True
except ImportError:
    _HAS_NICE = False

try:
    from .trade_and_conference import scan_trade_and_conferences as _trade_conf_scan
    _HAS_TRADE_CONF = True
except ImportError:
    _HAS_TRADE_CONF = False

try:
    from . import eudract_drug_trials as _eudract
except ImportError:
    _eudract = None  # type: ignore

try:
    from . import ctri_trials as _ctri
except ImportError:
    _ctri = None  # type: ignore

try:
    from .fetch_trial_ids import fetch_trial_ids as _fetch_intl_ids
except ImportError:
    _fetch_intl_ids = None  # type: ignore

try:
    from .pubmed_api import (
        search_pubmed as _pubmed_search,
        fetch_articles as _pubmed_fetch,
        extract_trial_ids as _pubmed_extract_trial_ids,
        articles_to_dataframe as _pubmed_articles_to_df,
    )
    import xml.etree.ElementTree as _ET
    import re as _re
    _HAS_PUBMED = True
except ImportError:
    _HAS_PUBMED = False

try:
    from .who import search_all_synonyms as _who_search
    _HAS_WHO = True
except ImportError:
    _HAS_WHO = False

try:
    from .alias_resolver import resolve_aliases as _resolve_aliases
    _HAS_ALIAS_RESOLVER = True
except ImportError:
    _HAS_ALIAS_RESOLVER = False

try:
    from .innovator_web import scan_innovator_website as _innovator_scan
    _HAS_INNOVATOR = True
except ImportError:
    _HAS_INNOVATOR = False

# ── GCP config from gcp_utils ────────────────────────────────────────────────
from .utils import (
    MODEL, get_gemini_client,
    alias_batch as _ALIAS_BATCH_SIZE,
    metadata_batch as _META_BATCH_SIZE,
)

from medical_potential.gcp_utils import get_bq_client
from medical_potential.config import GD_CLINICAL_TRIALS_FULL_TABLE_ID

try:
    from google import genai
    from google.genai import types as _gtypes
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

try:
    from .enrich_outcomes import enrich_trial_outcomes as _enrich
except ImportError:
    _enrich = None  # type: ignore

# ── output columns ─────────────────────────────────────────────────────────────
COLUMNS = [
    "molecule_name",
    "registry_source",
    "trial_id",
    "acronym",
    "dosage",
    "phase",
    "trial_title",
    "trial_study",
    "trial_size",
    "trial_location",
    "secondary_locations",
    "trial_start_date",
    "trial_completion_date",
    "phase_status",
    "hba1c_change_pct",
    "hba1c_duration",
    "hba1c_rationale",
    "hba1c_confidence",
    "weight_change_pct",
    "weight_duration",
    "weight_rationale",
    "weight_confidence",
    "alt_reduction_pct",
    "alt_duration",
    "alt_rationale",
    "alt_confidence",
    "mash_change_pct",
    "mash_duration",
    "mash_rationale",
    "mash_confidence",
    "company_name",
    "source_url",
    "llm_confidence",
    "llm_confidence_rationale",
    "eval_hba1c_confidence",
    "eval_weight_confidence",
    "eval_mash_confidence",
    "eval_alt_confidence",
    "eval_hba1c_pct_change",
    "eval_weight_pct_change",
    "eval_mash_pct_change",
    "eval_alt_pct_change",
    "trial_source",
]

# ── acronym fetch via ClinicalTrials.gov API ──────────────────────────────────

def _fetch_acronym_ctgov(trial_id: str, session: requests.Session) -> str:
    """
    Query the ClinicalTrials.gov v2 API for the acronym of a given NCT ID.
    Returns the acronym string or '' if not found.
    """
    if not trial_id or not trial_id.upper().startswith("NCT"):
        return ""
    url = f"https://clinicaltrials.gov/api/v2/studies/{trial_id}"
    try:
        resp = session.get(url, timeout=20)
        if not resp.ok:
            return ""
        data = resp.json()
        # path: protocolSection → identificationModule → acronym
        ps = data.get("protocolSection") or {}
        ident = ps.get("identificationModule") or {}
        return ident.get("acronym", "") or ""
    except Exception:
        return ""


def _fetch_acronym_euctr(trial_id: str, session: requests.Session) -> str:
    """
    Attempt to fetch an acronym for a CTIS trial via the public retrieve endpoint.
    """
    if not trial_id:
        return ""
    url = f"https://euclinicaltrials.eu/ctis-public-api/retrieve/{trial_id}"
    try:
        resp = session.get(url, timeout=20)
        if not resp.ok:
            return ""
        data = resp.json()
        # CTIS stores acronym in authorizedApplication → authorizedPartI → trialInformation
        p1 = (data.get("authorizedApplication") or {}).get("authorizedPartI") or {}
        trial_info = p1.get("trialInformation") or {}
        return trial_info.get("acronym") or trial_info.get("shortTitle") or ""
    except Exception:
        return ""


# ── per-registry mappers ───────────────────────────────────────────────────────

def _blank(molecule: str, source: str) -> Dict[str, str]:
    row: Dict[str, str] = {c: "" for c in COLUMNS}
    row["molecule_name"]   = molecule
    row["registry_source"] = source
    return row


def _split_locations(countries_str: str) -> tuple[str, str]:
    """
    Split a comma/semicolon-separated countries string into
    (primary_location, secondary_locations).

    Primary = first country listed.
    Secondary = remaining unique countries, excluding the primary, comma-separated.
    """
    if not countries_str or not countries_str.strip():
        return "", ""
    # Normalise separators
    parts = [c.strip() for c in countries_str.replace(";", ",").split(",") if c.strip()]
    if not parts:
        return "", ""
    primary = _clean_country_name(parts[0])
    # Dedupe and remove primary from secondary list
    seen: dict = {}
    for c in parts[1:]:
        c = _clean_country_name(c)
        if c and c.lower() != primary.lower():
            seen[c] = None   # dict preserves insertion order, dedupes by key
    secondary = ", ".join(seen.keys())
    return primary, secondary


# ── Data normalisation helpers ────────────────────────────────────────────────

import re as _re_norm

_COUNTRY_COLON_SUFFIX = _re_norm.compile(r'\s*:\s*\d+\s*$')

def _clean_country_name(name: str) -> str:
    """Remove trailing ':N' suffixes from country names.
    
    Examples:
        'Slovakia:2'  -> 'Slovakia'
        'Czechia:8'   -> 'Czechia'
        'United States: 15' -> 'United States'
    """
    if not name:
        return name
    return _COUNTRY_COLON_SUFFIX.sub('', name).strip()


def _normalise_location(loc_str: str) -> str:
    """Clean a location string: strip :N suffixes from each country name."""
    if not loc_str or not loc_str.strip():
        return loc_str
    parts = [c.strip() for c in loc_str.replace(";", ",").split(",") if c.strip()]
    cleaned = [_clean_country_name(p) for p in parts]
    return ", ".join(c for c in cleaned if c)


def _normalise_date(date_str: str) -> str:
    """Normalise a date string to YYYY-MM-DD format.
    
    Handles common formats:
        '01-01-2028'       -> '2028-01-01'
        '(01-01-2028)'     -> '2028-01-01'
        'January 1, 2028'  -> '2028-01-01'
        '2028-01-01'       -> '2028-01-01' (already correct)
        'March 2025'       -> '2025-03-01'
        '2025'             -> '2025'
    """
    if not date_str or not date_str.strip():
        return date_str
    s = date_str.strip().strip("()")

    # Already YYYY-MM-DD
    if _re_norm.match(r'^\d{4}-\d{2}-\d{2}$', s):
        return s

    # DD-MM-YYYY or DD/MM/YYYY
    m = _re_norm.match(r'^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$', s)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    # YYYY/MM/DD
    m = _re_norm.match(r'^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$', s)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"

    # Month DD, YYYY  or  Month YYYY
    _MONTHS = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12',
    }
    m = _re_norm.match(r'^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$', s)
    if m:
        month_name = m.group(1).lower()
        if month_name in _MONTHS:
            return f"{m.group(3)}-{_MONTHS[month_name]}-{m.group(2).zfill(2)}"

    # Month YYYY (no day)
    m = _re_norm.match(r'^([A-Za-z]+)\s+(\d{4})$', s)
    if m:
        month_name = m.group(1).lower()
        if month_name in _MONTHS:
            return f"{m.group(2)}-{_MONTHS[month_name]}-01"

    # DD Month YYYY
    m = _re_norm.match(r'^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$', s)
    if m:
        month_name = m.group(2).lower()
        if month_name in _MONTHS:
            return f"{m.group(3)}-{_MONTHS[month_name]}-{m.group(1).zfill(2)}"

    # Just a year
    m = _re_norm.match(r'^(\d{4})$', s)
    if m:
        return s

    # Could not parse — return stripped original
    return s


# Phase normalisation patterns
_PHASE_ROMAN = {'I': '1', 'II': '2', 'III': '3', 'IV': '4'}

def _normalise_phase(phase_str: str) -> str:
    """Normalise phase to standard format: Phase 1, Phase 2, Phase 3, Phase 4.
    
    Handles:
        'Phase III'                              -> 'Phase 3'
        'Phase 1/Phase 2'                        -> 'Phase 1/Phase 2'
        'PHASE 2'                                -> 'Phase 2'
        'phase2'                                 -> 'Phase 2'
        'Therapeutic exploratory (Phase III)'     -> 'Phase 3'
        'Phase 3a'                               -> 'Phase 3'
        'Phase 2b'                               -> 'Phase 2'
        'Phase 1b/2a'                            -> 'Phase 1/Phase 2'
        'Phase II/III'                           -> 'Phase 2/Phase 3'
        'N/A'                                    -> ''
        'Not Applicable'                         -> ''
        'Approved'                               -> 'Phase 4'
        'Post-marketing'                         -> 'Phase 4'
    """
    if not phase_str or not phase_str.strip():
        return ""
    
    s = phase_str.strip()
    
    # Empty-ish values
    if s.lower() in ('n/a', 'na', 'not applicable', 'not available', 'none', 'null', ''):
        return ""

    # Post-marketing / Approved = Phase 4
    if s.lower() in ('post-marketing', 'post marketing', 'approved', 'postmarketing'):
        return "Phase 4"

    # Extract phase info from strings like "Therapeutic exploratory (Phase III)"
    # or "Therapeutic confirmatory (Phase II)"
    paren_match = _re_norm.search(r'\(([^)]*phase[^)]*)\)', s, _re_norm.IGNORECASE)
    if paren_match:
        s = paren_match.group(1).strip()

    # Handle combined phases: Phase 1/Phase 2, Phase I/II, Phase 1b/2a, etc.
    # First try to find slash-separated phases
    slash_match = _re_norm.search(
        r'(?:phase\s*)?([IV]{1,3}|\d)[ab]?\s*/\s*(?:phase\s*)?([IV]{1,3}|\d)[ab]?',
        s, _re_norm.IGNORECASE
    )
    if slash_match:
        p1 = slash_match.group(1).upper()
        p2 = slash_match.group(2).upper()
        p1 = _PHASE_ROMAN.get(p1, p1)
        p2 = _PHASE_ROMAN.get(p2, p2)
        if p1.isdigit() and p2.isdigit():
            return f"Phase {p1}/Phase {p2}"

    # Single phase: "Phase III", "phase2", "Phase 3b", "PHASE IV", etc.
    single_match = _re_norm.search(
        r'phase\s*([IV]{1,3}|\d)[ab]?', s, _re_norm.IGNORECASE
    )
    if single_match:
        p = single_match.group(1).upper()
        p = _PHASE_ROMAN.get(p, p)
        if p.isdigit():
            return f"Phase {p}"

    # Bare roman numeral or digit (e.g. "III", "3", "2b")
    bare_match = _re_norm.match(r'^([IV]{1,3}|\d)[ab]?$', s.strip(), _re_norm.IGNORECASE)
    if bare_match:
        p = bare_match.group(1).upper()
        p = _PHASE_ROMAN.get(p, p)
        if p.isdigit():
            return f"Phase {p}"

    # Early Phase 1 / Phase 0
    if _re_norm.search(r'early\s*phase\s*1|phase\s*0', s, _re_norm.IGNORECASE):
        return "Phase 1"

    # Couldn't parse — return empty (not a recognisable phase)
    return ""


def _normalise_row(row: Dict[str, str]) -> None:
    """Apply all normalisations to a single row in-place."""
    # Location: strip :N suffixes
    if row.get("trial_location"):
        row["trial_location"] = _normalise_location(row["trial_location"])
    if row.get("secondary_locations"):
        row["secondary_locations"] = _normalise_location(row["secondary_locations"])

    # Dates: normalise to YYYY-MM-DD
    if row.get("trial_start_date"):
        row["trial_start_date"] = _normalise_date(row["trial_start_date"])
    if row.get("trial_completion_date"):
        row["trial_completion_date"] = _normalise_date(row["trial_completion_date"])

    # Phase: normalise to "Phase N" format
    if row.get("phase"):
        row["phase"] = _normalise_phase(row["phase"])


def map_ctgov(raw: Dict[str, Any], molecule: str,
              session: requests.Session) -> Dict[str, str]:
    row = _blank(molecule, "ClinicalTrials.gov")

    trial_id = raw.get("trial_id", "")
    row["trial_id"]            = trial_id
    row["acronym"]             = _fetch_acronym_ctgov(trial_id, session)
    row["phase"]               = raw.get("phase", "")
    row["trial_title"]         = raw.get("title", "") or raw.get("public_title", "")
    row["trial_study"]         = raw.get("study_type", "") + (
                                    f" | {raw.get('study_design','')}"
                                    if raw.get("study_design") else "")
    row["trial_size"]          = raw.get("actual_enrollment") or raw.get("target_enrollment", "")
    primary, secondary = _split_locations(raw.get("countries", ""))
    row["trial_location"]      = primary
    row["secondary_locations"] = secondary
    row["trial_start_date"]    = raw.get("start_date", "")
    row["trial_completion_date"] = raw.get("completion_date", "")
    row["phase_status"]        = raw.get("status", "")
    row["company_name"]        = raw.get("sponsor", "")
    row["source_url"]          = raw.get("url", "")
    # dosage intentionally left blank

    return row


def map_ctis(raw: Dict[str, Any], molecule: str,
             session: requests.Session) -> Dict[str, str]:
    row = _blank(molecule, "EU CTIS")

    trial_id = raw.get("ct_number", "") or raw.get("trial_id", "")
    row["trial_id"]              = trial_id
    row["acronym"]               = _fetch_acronym_euctr(trial_id, session)
    row["phase"]                 = raw.get("phase", "")
    row["trial_title"]           = raw.get("title", "") or raw.get("short_title", "")
    row["trial_study"]           = raw.get("trial_design", "")
    row["trial_size"]            = (raw.get("planned_subjects_worldwide", "")
                                    or raw.get("enrolled", ""))
    primary, secondary = _split_locations(raw.get("countries", ""))
    row["trial_location"]        = primary
    row["secondary_locations"]   = secondary
    row["trial_start_date"]      = raw.get("start_date", "")
    row["trial_completion_date"] = raw.get("end_date", "")
    row["phase_status"]          = raw.get("status", "")
    row["company_name"]          = raw.get("sponsors_full", "") or raw.get("sponsor", "")
    row["source_url"]            = raw.get("url", "")
    # dosage intentionally left blank

    return row


def map_eudract(raw: Dict[str, Any], molecule: str) -> Dict[str, str]:
    row = _blank(molecule, "EudraCT")

    trial_id = raw.get("eudract_number", "")
    row["trial_id"]              = trial_id
    # EudraCT has no dedicated acronym field; use sponsor protocol number as proxy
    row["acronym"]               = raw.get("sponsor_protocol_number", "")
    row["phase"]                 = raw.get("phase", "")
    row["trial_title"]           = (raw.get("full_title", "")
                                    or raw.get("title", "")
                                    or raw.get("lay_title", ""))
    design_parts = [raw.get("randomised", ""), raw.get("double_blind", ""),
                    raw.get("parallel_group", ""), raw.get("crossover", "")]
    row["trial_study"]           = " | ".join(p for p in design_parts if p)
    row["trial_size"]            = (raw.get("results_subjects_worldwide", "")
                                    or raw.get("subjects_worldwide", ""))
    primary, secondary = _split_locations(raw.get("countries", ""))
    row["trial_location"]        = primary
    row["secondary_locations"]   = secondary
    row["trial_start_date"]      = raw.get("start_date", "")
    row["trial_completion_date"] = raw.get("global_end_date", "")
    row["phase_status"]          = (raw.get("end_of_trial_status", "")
                                    or raw.get("status", ""))
    row["company_name"]          = (raw.get("sponsor_name", "")
                                    or raw.get("sponsor", ""))
    row["source_url"]            = raw.get("url", "") or raw.get("results_url", "")
    # dosage intentionally left blank

    return row


def map_ctri(raw: Dict[str, Any], molecule: str) -> Dict[str, str]:
    row = _blank(molecule, "CTRI (India)")

    row["trial_id"]              = raw.get("trial_id", "")
    row["acronym"]               = ""   # CTRI does not expose an acronym field
    row["phase"]                 = raw.get("phase", "")
    row["trial_title"]           = raw.get("title", "") or raw.get("public_title", "")
    row["trial_study"]           = raw.get("study_type", "") + (
                                      f" | {raw.get('study_design', '')}"
                                      if raw.get("study_design") else "")
    row["trial_size"]            = (raw.get("actual_enrollment", "")
                                    or raw.get("target_enrollment", ""))
    primary, secondary = _split_locations(raw.get("countries", "") or "India")
    row["trial_location"]        = primary or "India"
    row["secondary_locations"]   = secondary
    row["trial_start_date"]      = raw.get("start_date", "")
    row["trial_completion_date"] = raw.get("completion_date", "")
    row["phase_status"]          = raw.get("status", "")
    row["company_name"]          = raw.get("sponsor", "")
    row["source_url"]            = raw.get("url", "")
    # dosage intentionally left blank

    return row


# ── session factory ────────────────────────────────────────────────────────────

def _make_session(extra_headers: Optional[Dict[str, str]] = None) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/125.0 Safari/537.36"),
    })
    if extra_headers:
        s.headers.update(extra_headers)
    return s


# ── deduplication ──────────────────────────────────────────────────────────────

_SOURCE_PRIORITY = {
    "ClinicalTrials.gov":        10,
    "EU CTIS":                    9,
    "EudraCT":                    8,
    "CTRI":                       7,
    "PubMed":                     6,
    "NICE Technology Appraisals": 5,
    "Medical Conferences":        4,
    "Pharma Trade Publications":  3,
    "Innovator Website":          2,
    "GD Clinical Trials":         11,
}


def _row_richness(row: Dict[str, str]) -> int:
    """Score a row by how many non-empty fields it has."""
    return sum(1 for v in row.values() if v and str(v).strip() and str(v).strip().lower() != "n/a")


def _deduplicate_trials(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Deduplicate trials by trial_id (case-insensitive).

    When the same trial_id appears from multiple sources, keep the row with
    the richest data (most non-empty fields). Ties broken by source priority
    (ClinicalTrials.gov > EU CTIS > EudraCT > CTRI > PubMed > others).
    """
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        tid = (row.get("trial_id") or "").strip().upper()
        if not tid:
            continue
        groups.setdefault(tid, []).append(row)

    deduped: List[Dict[str, str]] = []
    for tid, group in groups.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            # Pick the richest row; break ties by source priority
            best = max(
                group,
                key=lambda r: (
                    _row_richness(r),
                    _SOURCE_PRIORITY.get(r.get("registry_source", ""), 0),
                ),
            )
            deduped.append(best)
    return deduped


# ── metadata backfill for bare rows ───────────────────────────────────────────

# _META_BATCH_SIZE is imported from utils (metadata_batch variable)
_META_MAX_WORKERS = 6

def _is_bare_row(row: Dict[str, str]) -> bool:
    """Return True if a row is missing basic registry metadata."""
    return not any([
        row.get("trial_title", "").strip(),
        row.get("company_name", "").strip(),
        row.get("trial_size", "").strip(),
    ])


def _build_metadata_prompt(molecule: str, batch: List[Dict[str, str]]) -> str:
    """Build a Gemini prompt to fetch registry metadata for a batch of trial IDs."""
    trial_lines = "\n".join(
        f"  - {r.get('trial_id', '?')} (source: {r.get('registry_source', '?')})"
        for r in batch
    )
    return f"""You are a clinical trial metadata extraction engine with access to Google Search.

MOLECULE: {molecule}

TRIAL IDs TO LOOK UP ({len(batch)} total):
{trial_lines}

For EACH trial ID above, search the relevant clinical trial registry
(ClinicalTrials.gov, ChiCTR, CTRI, JRCT, ANZCTR, CRIS, ReBEC, EU CTR,
IRCT, DRKS, NTR, PACTR, SLCTR, TCTR, WHO ICTRP, etc.)
and extract the following metadata:

- trial_id:              The trial ID exactly as given above
- trial_title:           Full official trial title from the registry
- phase:                 Trial phase (e.g. "Phase 3", "Phase 2", "Phase 1")
- trial_study:           Study type / design (e.g. "Interventional | Randomized, Double-blind")
- trial_size:            Total enrollment number (e.g. "1200")
- trial_location:        Primary country / region where the trial is led (e.g. "United States")
- secondary_locations:   All other countries where the trial is conducted, comma-separated (e.g. "Germany, Japan, India"). \"N/A\" if single-country.
- trial_start_date:      Study start date (YYYY-MM-DD or YYYY-MM)
- trial_completion_date: Primary completion date (YYYY-MM-DD or YYYY-MM)
- phase_status:          Current status (e.g. "Completed", "Recruiting", "Active, not recruiting")
- company_name:          Sponsor / lead organization
- source_url:            Direct URL to the trial on its registry

RULES:
- Return one JSON object per trial, keyed by trial_id
- Use "N/A" for genuinely unavailable fields — do NOT guess
- For NCT IDs, use ClinicalTrials.gov as primary source
- For non-NCT IDs, search the appropriate international registry

Return ONLY valid JSON, no markdown, no preamble:

{{
  "results": {{
    "<trial_id_1>": {{
      "trial_title": "...",
      "phase": "...",
      "trial_study": "...",
      "trial_size": "...",
      "trial_location": "...",
      "secondary_locations": "...",
      "trial_start_date": "...",
      "trial_completion_date": "...",
      "phase_status": "...",
      "company_name": "...",
      "source_url": "..."
    }}
  }}
}}
"""


async def _fetch_metadata_batch(
    molecule: str,
    batch: List[Dict[str, str]],
    batch_idx: int,
    total_batches: int,
    semaphore,
) -> Dict[str, Dict[str, str]]:
    """Fetch registry metadata for a batch of bare trial rows via Gemini+Search.
    
    Strategy: try with Google Search grounding first.  If the response cannot
    be parsed (often caused by TOO_MANY_TOOL_CALLS with non-NCT IDs from
    obscure registries), retry once WITHOUT search grounding so Gemini uses
    its training-data knowledge instead.
    """
    import asyncio as _aio

    async with semaphore:
        ids = [r.get("trial_id", "?") for r in batch]
        print(f"  [META] Batch {batch_idx+1}/{total_batches} -> {ids}", file=sys.stderr)

        prompt = _build_metadata_prompt(molecule, batch)

        try:
            if not _HAS_GENAI:
                print("  [META] google-genai not installed — skipping metadata backfill.", file=sys.stderr)
                return {}

            client = get_gemini_client()
            contents = [_gtypes.Content(role="user", parts=[_gtypes.Part.from_text(text=prompt)])]

            # --- Attempt 1: with Google Search grounding ---
            config_search = _gtypes.GenerateContentConfig(
                tools=[_gtypes.Tool(googleSearch=_gtypes.GoogleSearch())]
            )

            def _sync_search():
                out = ""
                for chunk in client.models.generate_content_stream(
                    model=MODEL, contents=contents, config=config_search
                ):
                    if chunk.text:
                        out += chunk.text
                return out.strip()

            raw = await _aio.to_thread(_sync_search)
            data = _parse_metadata_json(raw)

            if data:
                print(f"  [META] Batch {batch_idx+1}: got metadata for {len(data)} trial(s)",
                      file=sys.stderr)
                return data

            # --- Attempt 2: WITHOUT search grounding (training-data fallback) ---
            print(f"  [META] Batch {batch_idx+1}: search-grounded call failed to parse, "
                  f"retrying without grounding …", file=sys.stderr)
            if raw:
                print(f"  [META] Batch {batch_idx+1}: raw response (first 300 chars): "
                      f"{raw[:300]!r}", file=sys.stderr)

            config_plain = _gtypes.GenerateContentConfig()

            def _sync_plain():
                out = ""
                for chunk in client.models.generate_content_stream(
                    model=MODEL, contents=contents, config=config_plain
                ):
                    if chunk.text:
                        out += chunk.text
                return out.strip()

            raw2 = await _aio.to_thread(_sync_plain)
            data2 = _parse_metadata_json(raw2)

            if data2:
                print(f"  [META] Batch {batch_idx+1}: fallback got metadata for {len(data2)} trial(s)",
                      file=sys.stderr)
                return data2

            print(f"  [META] Batch {batch_idx+1}: could not parse response (both attempts)",
                  file=sys.stderr)
            if raw2:
                print(f"  [META] Batch {batch_idx+1}: fallback raw (first 300 chars): "
                      f"{raw2[:300]!r}", file=sys.stderr)
            return {}

        except Exception as exc:
            print(f"  [META] Gemini call failed: {exc}", file=sys.stderr)
            return {}


def _parse_metadata_json(raw: str) -> Dict[str, Dict[str, str]]:
    """Parse JSON from the metadata Gemini response.
    
    Handles several shapes Gemini may return:
      1. {"results": {"NCT...": {...}, ...}}          — expected format
      2. {"NCT...": {...}, "KCT...": {...}}            — top-level keyed by trial_id
      3. [{"trial_id": "NCT...", ...}, ...]            — list of dicts
      4. {"results": [{"trial_id": "NCT...", ...}]}    — results as list
    """
    if not raw:
        return {}
    text = raw.strip()
    # Strip markdown fences
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
    # Find first JSON structure
    for i, ch in enumerate(text):
        if ch in "{[":
            text = text[i:]
            break
    # Try parsing
    try:
        from json_repair import repair_json
        parsed = repair_json(text, return_objects=True)
    except Exception:
        try:
            parsed = json.loads(text)
        except Exception:
            return {}

    # Shape 1: {"results": {"trial_id": {...}, ...}}
    if isinstance(parsed, dict):
        results_val = parsed.get("results")
        if isinstance(results_val, dict):
            # Verify the values are dicts (not scalars)
            first_val = next(iter(results_val.values()), None) if results_val else None
            if isinstance(first_val, dict):
                return results_val

        # Shape 4: {"results": [{...}, {...}]}
        if isinstance(results_val, list):
            out: Dict[str, Dict[str, str]] = {}
            for item in results_val:
                if isinstance(item, dict) and item.get("trial_id"):
                    out[str(item["trial_id"]).strip()] = item
            if out:
                return out

        # Shape 2: top-level keys are trial IDs
        first_val = next(iter(parsed.values()), None) if parsed else None
        if isinstance(first_val, dict):
            return parsed

    # Shape 3: list of dicts with trial_id field
    if isinstance(parsed, list):
        out2: Dict[str, Dict[str, str]] = {}
        for item in parsed:
            if isinstance(item, dict) and item.get("trial_id"):
                out2[str(item["trial_id"]).strip()] = item
        if out2:
            return out2

    return {}


async def _backfill_metadata_and_locations(molecule: str, all_rows: List[Dict[str, str]]) -> None:
    """Backfill registry metadata AND secondary_locations in a single pass.
    
    Targets all rows that are bare (missing title/company/size) OR missing
    secondary_locations. Uses one Gemini+Search call per batch to fill both
    metadata fields and location fields together.
    """
    import asyncio as _aio

    # Target: any row that is bare OR missing secondary_locations
    targets = [
        r for r in all_rows
        if _is_bare_row(r) or not r.get("secondary_locations", "").strip()
    ]
    if not targets:
        print("[META+LOC] All rows already have metadata and locations — skipping.",
              file=sys.stderr)
        return

    print(f"\n[META+LOC] {len(targets)} row(s) need metadata/location backfill …",
          file=sys.stderr)

    batches = [targets[i:i+_META_BATCH_SIZE] for i in range(0, len(targets), _META_BATCH_SIZE)]
    total = len(batches)
    semaphore = _aio.Semaphore(_META_MAX_WORKERS)

    async def _staggered(coro, delay):
        await _aio.sleep(delay)
        return await coro

    tasks = [
        _staggered(
            _fetch_metadata_batch(molecule, batch, idx, total, semaphore),
            idx * 0.5
        )
        for idx, batch in enumerate(batches)
    ]
    results = await _aio.gather(*tasks)

    # Merge results back into target rows (in-place)
    merged: Dict[str, Dict[str, str]] = {}
    for r in results:
        if isinstance(r, dict):
            merged.update(r)

    filled = 0
    META_FIELDS = [
        "trial_title", "phase", "trial_study", "trial_size",
        "trial_location", "secondary_locations", "trial_start_date", "trial_completion_date",
        "phase_status", "company_name", "source_url",
    ]

    for row in targets:
        tid = row.get("trial_id", "").strip()
        # Try exact match, then case-insensitive
        meta = merged.get(tid)
        if not meta:
            for k, v in merged.items():
                if k.strip().upper() == tid.upper():
                    meta = v
                    break
        if not meta:
            continue
        for field in META_FIELDS:
            val = meta.get(field, "")
            if val and str(val).strip() and str(val).strip().lower() not in ("n/a", "null", "none"):
                if not row.get(field, "").strip():
                    row[field] = str(val).strip()
        filled += 1

    print(f"  [META+LOC] Backfilled metadata+locations for {filled}/{len(targets)} row(s).",
          file=sys.stderr)


# ── LLM fill for secondary_locations ─────────────────────────────────────────

_LOC_BATCH_SIZE  = 8    # trials per Gemini call
_LOC_MAX_WORKERS = 4    # concurrent Gemini calls


def _build_locations_prompt(molecule: str, batch: List[Dict[str, str]]) -> str:
    """Build a Gemini+Search prompt to fetch secondary_locations for a batch."""
    trial_lines = "\n".join(
        f"  - {r.get('trial_id','?')} | {r.get('trial_title','')[:100]} "
        f"| {r.get('registry_source','?')} | primary: {r.get('trial_location','unknown')}"
        for r in batch
    )
    return f"""You are a clinical trial data assistant with access to Google Search.

MOLECULE: {molecule}

TRIALS NEEDING COUNTRY DATA ({len(batch)} total):
{trial_lines}

For EACH trial above, find ALL countries where the trial is being or was conducted.
Search the registry page, ClinicalTrials.gov, the innovator's trial portal, and
any published results for each trial.

Return:
- trial_location:     The PRIMARY country (usually where the sponsor is headquartered
                      or where the most sites are). Use "Global" if it is a
                      multinational trial with no single dominant country.
- secondary_locations: All OTHER countries where the trial runs, comma-separated.
                       Use "" (empty string) if the trial is single-country.
                       Do NOT repeat the primary country here.

IMPORTANT:
- Look up EVERY trial — do not skip any.
- For multinational Phase 3 trials, secondary_locations may be 10–50 countries.
- For small Phase 1 trials, secondary_locations may be empty.
- Use standard English country names (e.g. "United States", "United Kingdom", "South Korea").

Return ONLY valid JSON, no markdown, no preamble:

{{
  "results": {{
    "<trial_id_1>": {{
      "trial_location": "United States",
      "secondary_locations": "Germany, Japan, India, Brazil, Canada"
    }},
    "<trial_id_2>": {{
      "trial_location": "Japan",
      "secondary_locations": ""
    }}
  }}
}}
"""


async def _fetch_locations_batch(
    molecule: str,
    batch: List[Dict[str, str]],
    batch_idx: int,
    total_batches: int,
    semaphore,
) -> Dict[str, Dict[str, str]]:
    """Fetch secondary_locations for a batch of trial rows via Gemini+Search."""
    import asyncio as _aio

    async with semaphore:
        ids = [r.get("trial_id", "?") for r in batch]
        print(f"  [LOC] Batch {batch_idx+1}/{total_batches} → {ids}", file=sys.stderr)

        if not _HAS_GENAI:
            print("  [LOC] google-genai not installed — skipping.", file=sys.stderr)
            return {}

        prompt = _build_locations_prompt(molecule, batch)
        client = get_gemini_client()
        contents = [_gtypes.Content(role="user", parts=[_gtypes.Part.from_text(text=prompt)])]
        config = _gtypes.GenerateContentConfig(
            tools=[_gtypes.Tool(googleSearch=_gtypes.GoogleSearch())]
        )

        def _sync_call():
            out = ""
            for chunk in client.models.generate_content_stream(
                model=MODEL, contents=contents, config=config
            ):
                if chunk.text:
                    out += chunk.text
            return out.strip()

        try:
            raw = await _aio.to_thread(_sync_call)
            data = _parse_metadata_json(raw)
            if data:
                print(f"  [LOC] Batch {batch_idx+1}: got locations for {len(data)} trial(s)",
                      file=sys.stderr)
                return data

            # Retry without search grounding
            config_plain = _gtypes.GenerateContentConfig()
            def _sync_plain():
                out = ""
                for chunk in client.models.generate_content_stream(
                    model=MODEL, contents=contents, config=config_plain
                ):
                    if chunk.text:
                        out += chunk.text
                return out.strip()

            raw2 = await _aio.to_thread(_sync_plain)
            data2 = _parse_metadata_json(raw2)
            if data2:
                print(f"  [LOC] Batch {batch_idx+1}: got locations (no-search fallback) "
                      f"for {len(data2)} trial(s)", file=sys.stderr)
                return data2

        except Exception as exc:
            print(f"  [LOC] Batch {batch_idx+1}: error — {exc}", file=sys.stderr)

    return {}


async def _fill_secondary_locations(molecule: str, rows: List[Dict[str, str]]) -> None:
    """
    For every row where secondary_locations is empty, use Gemini+Search to
    look up all countries the trial ran in and fill both trial_location and
    secondary_locations in-place.
    """
    import asyncio as _aio

    # Target all rows that are missing secondary_locations
    targets = [r for r in rows if not r.get("secondary_locations", "").strip()]
    if not targets:
        print("[LOC]   All rows already have secondary_locations — skipping.", file=sys.stderr)
        return

    print(f"\n[LOC]   {len(targets)} row(s) need secondary_locations …", file=sys.stderr)

    batches = [targets[i:i+_LOC_BATCH_SIZE] for i in range(0, len(targets), _LOC_BATCH_SIZE)]
    total = len(batches)
    semaphore = _aio.Semaphore(_LOC_MAX_WORKERS)

    async def _staggered(coro, delay):
        await _aio.sleep(delay)
        return await coro

    tasks = [
        _staggered(
            _fetch_locations_batch(molecule, batch, idx, total, semaphore),
            idx * 0.4,
        )
        for idx, batch in enumerate(batches)
    ]
    results = await _aio.gather(*tasks)

    # Merge all batch results
    merged: Dict[str, Dict[str, str]] = {}
    for r in results:
        if isinstance(r, dict):
            merged.update(r)

    filled = 0
    for row in targets:
        tid = row.get("trial_id", "").strip()
        meta = merged.get(tid) or merged.get(tid.upper())
        if not meta:
            for k, v in merged.items():
                if k.strip().upper() == tid.upper():
                    meta = v
                    break
        if not meta:
            continue
        # Fill trial_location if also missing
        loc = str(meta.get("trial_location", "")).strip()
        if loc and loc.lower() not in ("n/a", "none", "null") and not row.get("trial_location", "").strip():
            row["trial_location"] = loc
        # Fill secondary_locations
        sec = str(meta.get("secondary_locations", "")).strip()
        if sec and sec.lower() not in ("n/a", "none", "null"):
            row["secondary_locations"] = sec
        filled += 1

    print(f"  [LOC]  Filled secondary_locations for {filled}/{len(targets)} row(s).",
          file=sys.stderr)


# ── main fetch orchestration ───────────────────────────────────────────────────

def fetch_all(
    molecule: str,
    max_records: Optional[int] = None,
    search_terms: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Fetch clinical trials for a molecule across all sources.

    Parameters
    ----------
    molecule     : Primary drug name (used as the canonical label in output rows).
    max_records  : Optional cap on records per registry per term.
    search_terms : List of search terms to use (primary name + aliases).
                   If None or empty, falls back to [molecule].
                   Every source is searched once per term; results are
                   deduplicated across terms by trial_id.
    """
    if not search_terms:
        search_terms = [molecule]

    # Deduplicate terms while preserving order
    seen_terms: set = set()
    unique_terms: List[str] = []
    for t in search_terms:
        if t and t.strip() and t.strip().lower() not in seen_terms:
            seen_terms.add(t.strip().lower())
            unique_terms.append(t.strip())

    if len(unique_terms) > 1:
        print(f"\n=== Searching {len(unique_terms)} term(s) for '{molecule}': "
              f"{unique_terms} ===\n", file=sys.stderr)

    unified: List[Dict[str, str]] = []
    global_seen_ids: set = set()   # dedup across ALL terms

    # ── Shared helpers (thread-safe row accumulation) ─────────────────────
    _add_lock = __import__("threading").Lock()

    def _add(rows: List[Dict[str, str]], target_list: List[Dict[str, str]],
             is_ctgov: bool = False, trial_source: str = "") -> int:
        """Append rows not already seen globally; return count added.
        
        NCT IDs are only accepted from ClinicalTrials.gov (is_ctgov=True).
        Non-CTGOV sources that return NCT IDs are silently skipped.
        Thread-safe via _add_lock.
        """
        added = 0
        nct_skipped = 0
        with _add_lock:
            for r in rows:
                tid = r.get("trial_id", "").strip().upper()
                if not tid:
                    continue
                # Only accept NCT IDs from ClinicalTrials.gov
                if tid.startswith("NCT") and not is_ctgov:
                    nct_skipped += 1
                    continue
                if tid not in global_seen_ids:
                    global_seen_ids.add(tid)
                    # Always stamp with primary molecule name
                    r["molecule_name"] = molecule
                    # Stamp trial_source (which file/source fetched this trial)
                    if trial_source:
                        r["trial_source"] = trial_source
                    target_list.append(r)
                    added += 1
        if nct_skipped:
            print(f"         ({nct_skipped} NCT ID(s) skipped — only accepted from ClinicalTrials.gov)",
                  file=sys.stderr)
        return added

    # ── Source functions (accept term + target_list as params) ─────────────

    def _src_ctgov(term, target_list, log_skip=True):
        if _ctgov is None:
            if log_skip:
                print("[CTGOV] ctgov_trials.py not importable – skipping.", file=sys.stderr)
            return
        print(f"[CTGOV] Searching for '{term}' …", file=sys.stderr)
        try:
            session = _make_session({"Accept": "application/json"})
            ctgov_rows = _ctgov.fetch(term, max_records=max_records)
            print(f"[CTGOV] {len(ctgov_rows)} trial(s) found.", file=sys.stderr)
            mapped = []
            for raw in ctgov_rows:
                try:
                    mapped.append(map_ctgov(raw, molecule, session))
                except Exception as exc:
                    print(f"  [CTGOV] map error: {exc}", file=sys.stderr)
            _add(mapped, target_list, is_ctgov=True, trial_source="ctgov_trials")
        except Exception as exc:
            print(f"[CTGOV] fetch failed: {exc}", file=sys.stderr)

    def _src_ctis(term, target_list, log_skip=True):
        if _ctis is None:
            if log_skip:
                print("[CTIS]  ctis_drug_trials.py not importable – skipping.", file=sys.stderr)
            return
        print(f"[CTIS]  Searching for '{term}' …", file=sys.stderr)
        try:
            session = _make_session(_ctis.HEADERS)
            ctis_rows: List[Dict[str, Any]] = []
            for summary in _ctis.search_trials(term, session,
                                               page_size=50,
                                               max_records=max_records):
                try:
                    details = _ctis.get_trial_details(
                        summary.get("ctNumber", ""), session)
                except Exception:
                    details = None
                ctis_rows.append(_ctis.flatten(summary, details))
                time.sleep(0.3)
            print(f"[CTIS]  {len(ctis_rows)} trial(s) found.", file=sys.stderr)
            mapped = []
            for raw in ctis_rows:
                try:
                    mapped.append(map_ctis(raw, molecule, session))
                except Exception as exc:
                    print(f"  [CTIS] map error: {exc}", file=sys.stderr)
            _add(mapped, target_list, trial_source="ctis_drug_trials")
        except Exception as exc:
            print(f"[CTIS]  fetch failed: {exc}", file=sys.stderr)

    def _src_eudract(term, target_list, log_skip=True):
        if _eudract is None:
            if log_skip:
                print("[EUCT]  eudract_drug_trials.py not importable – skipping.", file=sys.stderr)
            return
        print(f"[EUCT]  Searching EudraCT for '{term}' …", file=sys.stderr)
        try:
            session = _make_session(_eudract.HEADERS)
            eudract_rows: List[Dict[str, Any]] = []
            for row in _eudract.search_trials(term, session,
                                              max_records=max_records):
                country = (row.get("countries", "").split(";")[0] or "GB").strip()
                try:
                    row.update(_eudract.get_trial_details(
                        row["eudract_number"], country, session))
                except Exception:
                    pass
                if row.get("results_available") == "Yes":
                    try:
                        row.update(_eudract.get_trial_results(
                            row["eudract_number"], session))
                    except Exception:
                        pass
                eudract_rows.append(row)
                time.sleep(_eudract.POLITE_DELAY)
            print(f"[EUCT]  {len(eudract_rows)} trial(s) found.", file=sys.stderr)
            mapped = []
            for raw in eudract_rows:
                try:
                    mapped.append(map_eudract(raw, molecule))
                except Exception as exc:
                    print(f"  [EUCT] map error: {exc}", file=sys.stderr)
            _add(mapped, target_list, trial_source="eudract_drug_trials")
        except Exception as exc:
            print(f"[EUCT]  fetch failed: {exc}", file=sys.stderr)

    def _src_ctri(term, target_list, log_skip=True):
        if _ctri is None:
            if log_skip:
                print("[CTRI]  ctri_trials.py not importable – skipping.", file=sys.stderr)
            return
        print(f"[CTRI]  Searching CTRI for '{term}' …", file=sys.stderr)
        try:
            ctri_rows = _ctri.fetch(term, max_records=max_records)
            print(f"[CTRI]  {len(ctri_rows)} trial(s) found.", file=sys.stderr)
            mapped = []
            for raw in ctri_rows:
                try:
                    mapped.append(map_ctri(raw, molecule))
                except Exception as exc:
                    print(f"  [CTRI] map error: {exc}", file=sys.stderr)
            _add(mapped, target_list, trial_source="ctri_trials")
        except Exception as exc:
            print(f"[CTRI]  fetch failed: {exc}", file=sys.stderr)

    def _src_pubmed(term, target_list, log_skip=True):
        if not _HAS_PUBMED:
            if log_skip:
                print("[PUBMED] pubmed_api.py not importable – skipping.", file=sys.stderr)
            return
        print(f"[PUBMED] Searching PubMed for '{term}' trial IDs …", file=sys.stderr)
        try:
            pmids = _pubmed_search(term)
            if pmids:
                articles    = _pubmed_fetch(pmids)
                pubmed_ids: set = set()
                non_nct_ids: List[Dict[str, str]] = []
                nct_pattern = _re.compile(r'NCT\d{8}', _re.IGNORECASE)

                for art_elem in articles:
                    abstract_parts = []
                    mc  = art_elem.find("MedlineCitation")
                    art = mc.find("Article") if mc is not None else None
                    if art is not None:
                        for ab in art.findall(".//AbstractText"):
                            if ab.text:
                                abstract_parts.append(ab.text)
                    abstract_text = " ".join(abstract_parts)

                    for db in art_elem.findall(".//DataBank"):
                        db_name = (db.findtext("DataBankName") or "").strip()
                        if "ClinicalTrials" in db_name:
                            for acc in db.findall(".//AccessionNumber"):
                                if acc.text and acc.text.strip():
                                    pubmed_ids.add(acc.text.strip().upper())
                    for m in nct_pattern.findall(abstract_text):
                        pubmed_ids.add(m.upper())
                    try:
                        trial_id_str, source_registry_str = _pubmed_extract_trial_ids(art_elem, abstract_text)
                        if trial_id_str:
                            for tid in trial_id_str.split(","):
                                tid = tid.strip().upper()
                                if tid and not tid.startswith("NCT"):
                                    non_nct_ids.append({
                                        "trial_id": tid,
                                        "registry_source": source_registry_str or "PubMed",
                                    })
                    except Exception:
                        pass

                pm_mapped = []
                for nct in sorted(pubmed_ids):
                    row = _blank(molecule, "PubMed")
                    row["trial_id"] = nct
                    pm_mapped.append(row)
                for item in non_nct_ids:
                    row = _blank(molecule, item["registry_source"])
                    row["trial_id"] = item["trial_id"]
                    pm_mapped.append(row)
                new_ncts    = _add(pm_mapped, target_list, trial_source="pubmed_api")
                print(f"[PUBMED] {len(pubmed_ids)} NCT ID(s) found, "
                      f"{new_ncts} new trial ID(s) added.", file=sys.stderr)
            else:
                print("[PUBMED] No PubMed results.", file=sys.stderr)
        except Exception as exc:
            print(f"[PUBMED] fetch failed: {exc}", file=sys.stderr)

    def _src_who(term, target_list, log_skip=True):
        if not _HAS_WHO:
            if log_skip:
                print("[WHO]   who.py not importable – skipping.", file=sys.stderr)
            return
        print(f"[WHO]   Searching WHO ICTRP for '{term}' …", file=sys.stderr)
        try:
            who_trials = _who_search([term])
            who_mapped = []
            for trial in who_trials:
                tid = str(trial.get("trial_id") or "").strip().upper()
                if not tid:
                    continue
                row = _blank(molecule, trial.get("source_registry", "WHO ICTRP"))
                row["trial_id"]         = tid
                row["trial_title"]      = str(trial.get("title") or "")
                row["phase_status"]     = str(trial.get("status") or "")
                row["trial_start_date"] = str(trial.get("date_registered") or "")
                row["source_url"]       = str(trial.get("detail_url") or "")
                row["phase"]            = str(trial.get("trial_phase") or "")
                who_mapped.append(row)
            added = _add(who_mapped, target_list, trial_source="who")
            print(f"[WHO]   {added} new trial(s) added from WHO ICTRP.", file=sys.stderr)
        except Exception as exc:
            print(f"[WHO]   fetch failed: {exc}", file=sys.stderr)

    def _src_intl(term, target_list, log_skip=True):
        if _fetch_intl_ids is None:
            if log_skip:
                print("[INTL]  fetch_trial_ids.py not importable – skipping.", file=sys.stderr)
            return
        print(f"[INTL]  Searching international registries for '{term}' …", file=sys.stderr)
        try:
            import asyncio as _asyncio
            intl_trials = _asyncio.run(_fetch_intl_ids(term))
            intl_mapped = []
            for trial in intl_trials:
                trial_id = trial.get("NCT_ID", "").strip()
                if not trial_id:
                    continue
                row = _blank(molecule, trial.get("registry_source", "International"))
                row["trial_id"]   = trial_id
                row["phase"]      = f"Phase {trial.get('Phase', '')}".strip()
                row["acronym"]    = trial.get("Program_Name", "") or ""
                row["trial_title"] = trial.get("Title", "") or ""
                # Preserve indication in trial_study if available
                indication = trial.get("Indication", "") or ""
                if indication and indication != "N/A":
                    row["trial_study"] = indication
                row["source_url"] = ""
                intl_mapped.append(row)
            added = _add(intl_mapped, target_list, trial_source="fetch_trial_ids")
            print(f"[INTL]  {added} trial(s) added from international registries.", file=sys.stderr)
        except Exception as exc:
            print(f"[INTL]  fetch failed: {exc}", file=sys.stderr)

    def _src_nice(term, target_list, log_skip=True):
        if not _HAS_NICE:
            if log_skip:
                print("[NICE]  nice_trials.py not importable – skipping.", file=sys.stderr)
            return
        print(f"[NICE]  Scanning NICE Technology Appraisals for '{term}' …",
              file=sys.stderr)
        try:
            nice_trials = _nice_scan(term)
            nice_mapped = []
            for item in nice_trials:
                tid = str(item.get("trial_id") or "").strip().upper()
                if not tid:
                    continue
                row = _blank(molecule, item.get("registry_source", "NICE Technology Appraisals"))
                row["trial_id"]       = tid
                row["trial_title"]    = str(item.get("trial_title") or "")[:300]
                row["phase"]          = str(item.get("phase") or "")
                row["phase_status"]   = str(item.get("phase_status") or "")
                row["trial_location"] = str(item.get("trial_location") or "UK")
                row["company_name"]   = str(item.get("company_name") or "")
                row["source_url"]     = str(item.get("source_url") or "")
                nice_mapped.append(row)
            added = _add(nice_mapped, target_list, trial_source="nice_trials")
            print(f"[NICE]  {added} new ID(s) added from NICE.", file=sys.stderr)
        except Exception as exc:
            print(f"[NICE]  fetch failed: {exc}", file=sys.stderr)

    def _src_trade_conf(term, target_list, log_skip=True):
        if not _HAS_TRADE_CONF:
            if log_skip:
                print("[TRADE&CONF] trade_and_conference.py not importable – skipping.", file=sys.stderr)
            return
        print(f"[TRADE&CONF] Scanning conferences + trade publications for '{term}' …",
              file=sys.stderr)
        try:
            tc_trials = _trade_conf_scan(term)
            tc_mapped = []
            for item in tc_trials:
                tid = str(item.get("trial_id") or "").strip().upper()
                if not tid:
                    continue
                src_type = str(item.get("source_type") or "")
                registry_label = (
                    "Medical Conferences" if src_type == "Conference"
                    else "Pharma Trade Publications"
                )
                row = _blank(molecule, item.get("registry_source", registry_label))
                row["trial_id"]    = tid
                row["trial_title"] = str(item.get("trial_title") or "")[:300]
                row["phase"]       = str(item.get("phase") or "")
                row["company_name"] = str(item.get("company_name") or "")
                row["source_url"]  = str(item.get("source_url") or "")
                tc_mapped.append(row)
            added = _add(tc_mapped, target_list, trial_source="trade_and_conference")
            print(f"[TRADE&CONF] {added} new ID(s) added.", file=sys.stderr)
        except Exception as exc:
            print(f"[TRADE&CONF] fetch failed: {exc}", file=sys.stderr)

    def _src_innovator(term, target_list, log_skip=True):
        if not _HAS_INNOVATOR:
            if log_skip:
                print("[INNOV] innovator_web.py not importable – skipping.", file=sys.stderr)
            return
        print(f"[INNOV] Scanning innovator website for '{term}' trial IDs …",
              file=sys.stderr)
        try:
            innovator_trials = _innovator_scan(term)
            innov_mapped = []
            for item in innovator_trials:
                tid = str(item.get("trial_id") or "").strip().upper()
                if not tid:
                    continue
                row = _blank(molecule, item.get("registry_source", "Innovator Website"))
                row["trial_id"]    = tid
                row["trial_title"] = str(item.get("trial_title") or "")[:300]
                row["phase"]       = str(item.get("phase") or "")
                row["company_name"] = str(item.get("company_name") or "")
                row["source_url"]  = str(item.get("source_url") or "")
                innov_mapped.append(row)
            added = _add(innov_mapped, target_list, trial_source="innovator_web")
            print(f"[INNOV] {added} new trial ID(s) added from innovator website.",
                  file=sys.stderr)
        except Exception as exc:
            print(f"[INNOV] fetch failed: {exc}", file=sys.stderr)

    def _src_gd_table(term, target_list, log_skip=True):
        """Source 11: GD clinical trials table from cognito-prod BigQuery."""
        print(f"[GD]    Searching GD table for '{term}' …", file=sys.stderr)
        try:
            bq = get_bq_client()

            query = f"""
                SELECT
                    Trial_Title, Trial_Status, Trial_Phase,
                    Cleaned_Drug_Name, Sponsor_Name, Primary_IDs,
                    Study_Type, Trial_Start_Date, Trial_End_Date,
                    Study_Design, Total_Count
                FROM `{GD_CLINICAL_TRIALS_FULL_TABLE_ID}`
                WHERE LOWER(Cleaned_Drug_Name) LIKE LOWER(@term)
            """
            from google.cloud import bigquery as _bq_mod
            job_config = _bq_mod.QueryJobConfig(
                query_parameters=[
                    _bq_mod.ScalarQueryParameter("term", "STRING", f"%{term}%")
                ]
            )
            result = bq.query(query, job_config=job_config).result()
            gd_mapped = []
            for bq_row in result:
                tid = str(bq_row.Primary_IDs or "").strip()
                if not tid:
                    continue
                # A row may have multiple comma-separated IDs
                for single_id in tid.split(","):
                    single_id = single_id.strip().upper()
                    if not single_id:
                        continue
                    row = _blank(molecule, "GD Clinical Trials")
                    row["trial_id"]              = single_id
                    row["trial_title"]           = str(bq_row.Trial_Title or "")[:300]
                    row["phase"]                 = str(bq_row.Trial_Phase or "")
                    row["trial_study"]           = str(bq_row.Study_Design or "")
                    row["company_name"]          = str(bq_row.Sponsor_Name or "")
                    row["trial_start_date"]      = str(bq_row.Trial_Start_Date or "")
                    row["trial_completion_date"] = str(bq_row.Trial_End_Date or "")
                    row["phase_status"]          = str(bq_row.Trial_Status or "")
                    row["trial_size"]            = str(bq_row.Total_Count or "")
                    row["llm_confidence"]        = "1"
                    row["llm_confidence_rationale"] = (
                        "Auto-assigned: trial sourced from curated GD Clinical Trials table."
                    )
                    gd_mapped.append(row)
            added = _add(gd_mapped, target_list, trial_source="ctgov_port")
            print(f"[GD]    {added} new trial ID(s) added from GD table.", file=sys.stderr)
        except Exception as exc:
            print(f"[GD]    fetch failed: {exc}", file=sys.stderr)

    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed_src

    # ══════════════════════════════════════════════════════════════════════
    # PHASE A: Run primary drug (first term) through ALL sources fully
    # ══════════════════════════════════════════════════════════════════════

    primary_term = unique_terms[0]
    alias_terms = unique_terms[1:]  # may be empty

    print(f"\n--- PRIMARY DRUG: '{primary_term}' (full pipeline) ---",
          file=sys.stderr)

    primary_rows: List[Dict[str, str]] = []

    all_sources = [
        ("CTGOV",      _src_ctgov),
        ("CTIS",       _src_ctis),
        ("EudraCT",    _src_eudract),
        ("CTRI",       _src_ctri),
        ("PubMed",     _src_pubmed),
        ("WHO",        _src_who),
        ("INTL",       _src_intl),
        ("NICE",       _src_nice),
        ("TRADE&CONF", _src_trade_conf),
        ("INNOV",      _src_innovator),
        ("GD",         _src_gd_table),
    ]

    with ThreadPoolExecutor(max_workers=10, thread_name_prefix="src") as src_pool:
        futures = {
            src_pool.submit(fn, primary_term, primary_rows, True): name
            for name, fn in all_sources
        }
        for fut in _as_completed_src(futures):
            name = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"[{name}] Unexpected error: {exc}", file=sys.stderr)

    unified.extend(primary_rows)
    print(f"\n[PRIMARY] {len(primary_rows)} trial(s) from primary drug '{primary_term}'.",
          file=sys.stderr)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE B: Run alias terms in batches of 4.
    #
    # Strategy:
    #   - API sources (CTGOV, CTIS, EudraCT, CTRI, PubMed, WHO, GD):
    #     Each API only accepts one search term at a time, so we call each
    #     API source once per alias — but run ALL aliases × ALL API sources
    #     in parallel in a single thread pool.
    #   - Gemini sources (INTL, NICE, TRADE&CONF, INNOV):
    #     These use LLM prompts. We pass a combined search string
    #     "alias1 OR alias2 OR alias3 OR alias4" so the LLM searches for
    #     up to 4 aliases in a SINGLE call per source per batch.
    # ══════════════════════════════════════════════════════════════════════

    # _ALIAS_BATCH_SIZE is imported from utils (alias_batch variable)

    if alias_terms:
        # Split aliases into batches of 4
        alias_batches = [alias_terms[i:i+_ALIAS_BATCH_SIZE]
                         for i in range(0, len(alias_terms), _ALIAS_BATCH_SIZE)]

        print(f"\n--- ALIASES: {alias_terms} ({len(alias_batches)} batch(es) of up to "
              f"{_ALIAS_BATCH_SIZE}) ---", file=sys.stderr)

        alias_rows: List[Dict[str, str]] = []

        # API sources: called once per alias, all in parallel
        api_sources = [
            ("CTGOV",   _src_ctgov),
            ("CTIS",    _src_ctis),
            ("EudraCT", _src_eudract),
            ("CTRI",    _src_ctri),
            ("PubMed",  _src_pubmed),
            ("WHO",     _src_who),
            ("GD",      _src_gd_table),
        ]

        # Gemini sources: called ONCE per batch with up to 4 aliases combined
        gemini_sources = [
            ("INTL",       _src_intl),
            ("NICE",       _src_nice),
            ("TRADE&CONF", _src_trade_conf),
            ("INNOV",      _src_innovator),
        ]

        for batch_idx, alias_batch in enumerate(alias_batches, 1):
            print(f"\n  [ALIAS BATCH {batch_idx}/{len(alias_batches)}] "
                  f"Processing: {alias_batch}", file=sys.stderr)

            # Build combined search term for Gemini sources:
            # e.g. "Ozempic OR Wegovy OR Rybelsus OR Saxenda"
            combined_alias_term = " OR ".join(alias_batch)

            # Submit ALL jobs into one thread pool:
            #   - (alias × api_source) pairs for this batch
            #   - (combined_alias_term × gemini_source) pairs
            with ThreadPoolExecutor(max_workers=14, thread_name_prefix="alias") as alias_pool:
                futures = {}

                # API sources: one call per alias per source
                for alias in alias_batch:
                    for src_name, src_fn in api_sources:
                        future = alias_pool.submit(src_fn, alias, alias_rows, False)
                        futures[future] = f"{src_name}({alias})"

                # Gemini sources: ONE call with combined aliases (up to 4)
                for src_name, src_fn in gemini_sources:
                    future = alias_pool.submit(src_fn, combined_alias_term, alias_rows, False)
                    futures[future] = f"{src_name}(batch-{batch_idx})"

                for fut in _as_completed_src(futures):
                    name = futures[fut]
                    try:
                        fut.result()
                    except Exception as exc:
                        print(f"[{name}] Unexpected error: {exc}", file=sys.stderr)

        unified.extend(alias_rows)
        print(f"\n[ALIASES] {len(alias_rows)} additional trial(s) from "
              f"{len(alias_terms)} alias(es).", file=sys.stderr)

    # ── end of primary + alias processing ─────────────────────────────────

    # ── Deduplication across all sources ──────────────────────────────────────
    before = len(unified)
    unified = _deduplicate_trials(unified)
    after = len(unified)
    if before != after:
        print(f"\n[DEDUP] {before} → {after} trial(s) after deduplication "
              f"({before - after} duplicate(s) removed).", file=sys.stderr)

    # ── Combined metadata + location backfill ───────────────────────────────
    try:
        import asyncio as _asyncio_meta
        _asyncio_meta.run(_backfill_metadata_and_locations(molecule, unified))
    except Exception as exc:
        print(f"[META+LOC] Combined backfill failed: {exc}", file=sys.stderr)

    # ── LLM Confidence Check — verify each trial belongs to this drug ─────────
    try:
        from .llm_eval import evaluate_trials as _llm_evaluate
        # Build alias string from search_terms
        alias_str = ", ".join(unique_terms) if len(unique_terms) > 1 else molecule
        _llm_evaluate(molecule, unified, aliases=alias_str)
    except ImportError:
        print("[LLM_EVAL] llm_eval.py not importable — skipping confidence check.",
              file=sys.stderr)
    except Exception as exc:
        print(f"[LLM_EVAL] Confidence check failed: {exc}", file=sys.stderr)

    # ── Final normalisation pass ──────────────────────────────────────────────
    # Clean locations (strip :N suffixes), normalise dates (YYYY-MM-DD),
    # and normalise phases (Phase 1/2/3/4 format).
    for row in unified:
        _normalise_row(row)
    print(f"[NORM] Normalised locations, dates, and phases for {len(unified)} row(s).",
          file=sys.stderr)

    return unified


# ── JSON writer ────────────────────────────────────────────────────────────────

def write_json(rows: List[Dict[str, str]], path: str) -> None:
    """Write the unified rows to a JSON file (one dict per trial)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(rows)} trial(s) → {path}", file=sys.stderr)


# ── top-N ranking ─────────────────────────────────────────────────────────────

def _rank_and_trim(rows: List[Dict[str, str]], top_n: int) -> List[Dict[str, str]]:
    """
    Score each trial by data completeness and return the top N.

    Scoring favours trials that have:
      - a recognised phase (Phase 3 > Phase 2 > Phase 1 > other)
      - an enrollment number (larger is better, capped)
      - start and completion dates
      - an acronym (published trials almost always have one)
      - a company name
    """

    def _phase_score(phase: str) -> int:
        p = (phase or "").lower()
        if "3" in p or "iii" in p:
            return 40
        if "4" in p or "iv" in p:
            return 35
        if "2" in p or "ii" in p:
            return 30
        if "1" in p or "i" in p:
            return 15
        return 0

    def _enrollment_score(size: str) -> int:
        try:
            n = int(str(size).replace(",", "").strip())
            return min(30, n // 10)           # up to 30 pts
        except (ValueError, TypeError):
            return 0

    def _score(row: Dict[str, str]) -> int:
        s = 0
        s += _phase_score(row.get("phase", ""))
        s += _enrollment_score(row.get("trial_size", ""))
        if row.get("trial_start_date"):
            s += 5
        if row.get("trial_completion_date"):
            s += 5
        if row.get("acronym"):
            s += 10
        if row.get("company_name"):
            s += 5
        if row.get("phase_status", "").lower() in ("completed", "active, not recruiting"):
            s += 10
        return s

    scored = sorted(rows, key=_score, reverse=True)
    return scored[:top_n]


# ── Public API ─────────────────────────────────────────────────────────────────

async def fetch_trials(
    molecule: str,
    max_records: Optional[int] = None,
    top_n: Optional[int] = None,
    no_enrich: bool = False,
    workers: int = 6,
    out_json: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Fetch clinical trials for a molecule across all registries.

    Parameters
    ----------
    molecule    : Primary drug name.
    max_records : Optional cap on records per registry.
    top_n       : Keep only the top N trials by completeness before enrichment.
    no_enrich   : Skip outcome enrichment when True.
    workers     : Concurrent workers for enrichment.
    out_json    : If provided, write results to this JSON file path.
                  Typically None — rows are returned in-memory to the caller.

    Returns
    -------
    List of trial dicts (one per trial).
    """
    molecule = molecule.strip()
    print(f"\n=== Fetching clinical trials for: {molecule} ===\n", file=sys.stderr)

    if _HAS_ALIAS_RESOLVER:
        search_terms = _resolve_aliases(molecule)
    else:
        print("[ALIAS] alias_resolver.py not importable — searching primary name only.",
              file=sys.stderr)
        search_terms = [molecule]

    # fetch_all is synchronous and contains some internal asyncio.run() calls
    # (which must not be invoked from an already-running event loop). Run
    # the blocking fetch_all in a thread so any internal asyncio.run() calls
    # execute in that separate thread safely.
    rows = await _asyncio.to_thread(fetch_all, molecule, max_records, search_terms)

    if not rows:
        print("No trials found across all registries.", file=sys.stderr)
        return []

    print(f"\nTotal trials collected: {len(rows)}", file=sys.stderr)

    if top_n and top_n > 0 and len(rows) > top_n:
        rows = _rank_and_trim(rows, top_n)
        print(f"Trimmed to top {top_n} trial(s) by completeness.", file=sys.stderr)

    if no_enrich:
        print("[ENRICH] no_enrich=True – skipping.", file=sys.stderr)
    elif _enrich is not None:
        rows = _enrich(rows, molecule, max_workers=workers)
    else:
        print("[ENRICH] enrich_outcomes.py not importable – skipping.", file=sys.stderr)

    # Only write JSON if explicitly requested (typically None in pipeline mode)
    if out_json:
        write_json(rows, out_json)

    return rows
