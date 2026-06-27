"""Field Extractor - combines layout graph and VLM output into field candidates."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _ocr_contains_value(ocr_blocks: List[Dict[str, Any]], value: str, threshold: float = 0.75) -> bool:
    """Check if a VLM-extracted value appears in any raw OCR block text.
    
    Uses substring and fuzzy-ish matching to handle minor OCR transcription differences.
    Returns True if the value (or a significant part of it) is found in OCR output.
    """
    if not value or not ocr_blocks:
        return False
    
    value_clean = re.sub(r"[\s\-\.]", "", value.lower())  # Remove spaces, dashes, dots
    if len(value_clean) < 3:
        return False  # Too short to be meaningful
    
    for block in ocr_blocks:
        block_text = re.sub(r"[\s\-\.]", "", str(block.get("text", "")).lower())
        if not block_text:
            continue
        # Direct substring match
        if value_clean in block_text or block_text in value_clean:
            return True
        # Partial match for longer values (>= 75% overlap)
        if len(value_clean) >= 6:
            # Check if most chars match (handles minor OCR errors)
            shorter = min(len(value_clean), len(block_text))
            longer = max(len(value_clean), len(block_text))
            if shorter / longer >= threshold and (
                value_clean[:shorter] == block_text[:shorter] or
                value_clean in block_text or block_text in value_clean
            ):
                return True
    return False


def normalize_field_name(raw_name: str) -> str:
    """Normalize a field name to snake_case canonical form."""
    name = raw_name.strip().rstrip(":").strip()
    # Remove common noise
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name.lower()).strip("_")
    return name if name else "unknown_field"


def infer_field_type(field_name: str, value: str) -> str:
    """Infer the likely type of a field from name and value."""
    name_lower = field_name.lower()

    # Date patterns
    if any(k in name_lower for k in ("date", "dob", "birth", "expiry", "issued")):
        return "date"

    # Phone/mobile
    if any(k in name_lower for k in ("phone", "mobile", "cell", "fax", "tel")):
        return "phone"

    # ID numbers
    if any(k in name_lower for k in ("cnic", "nic", "passport", "id_no", "national")):
        return "national_id"

    # Email
    if "email" in name_lower or "@" in value:
        return "email"

    # Account/amount
    if "title" in name_lower:
        return "text"
    if any(k in name_lower for k in ("account_number", "account_no", "acc_no", "iban")):
        return "account_number"
    if any(k in name_lower for k in ("amount", "balance", "deposit", "total")):
        return "amount"

    # Name
    if any(k in name_lower for k in ("name", "applicant", "nominee", "father", "mother", "husband")):
        return "person_name"

    # Address
    if any(k in name_lower for k in ("address", "city", "town", "province", "state", "zip", "postal")):
        return "address"

    # Checkbox
    if any(k in name_lower for k in ("checkbox", "check")):
        return "checkbox"
    if value.lower() in ("checked", "unchecked", "yes", "no"):
        return "checkbox"

    # Signature
    if "signature" in name_lower:
        return "signature"

    return "text"


def extract_fields_from_graph(
    layout_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract field candidates from the layout graph using label-value edges.
    
    Returns list of field candidates with metadata.
    """
    nodes_by_id = {n["node_id"]: n for n in layout_graph.get("nodes", [])}
    edges = layout_graph.get("edges", [])
    candidates: List[Dict[str, Any]] = []

    for edge in edges:
        if edge["edge_type"] not in ("right_of", "below", "likely_value_for"):
            continue

        label_node = nodes_by_id.get(edge["from"])
        value_node = nodes_by_id.get(edge["to"])

        if not label_node or not value_node:
            continue

        # Skip if label is actually a form header/noise
        if label_node.get("node_type") in ("header", "noise"):
            continue

        # Skip if value node is actually a label (label→label false pairing)
        if value_node.get("node_type") == "label":
            continue

        field_name_raw = label_node["text"]
        field_name_normalized = normalize_field_name(field_name_raw)
        value_raw = value_node["text"]

        # Skip clearly noisy field names (very short, only symbols, or all-caps header style)
        if not field_name_normalized or field_name_normalized == "unknown_field":
            continue
        if len(field_name_raw.strip()) < 3:
            continue

        field_type = infer_field_type(field_name_normalized, value_raw)

        candidate = {
            "field_name_raw": field_name_raw,
            "field_name_normalized": field_name_normalized,
            "value_raw": value_raw,
            "value_normalized": value_raw.strip(),
            "field_type": field_type,
            "label_bbox": label_node["bbox"],
            "value_bbox": value_node["bbox"],
            "geometry_score": edge["score"],
            "source_models": list(set(
                label_node.get("source_models", []) + value_node.get("source_models", [])
            )),
            "label_confidence": label_node.get("confidence", 0),
            "value_confidence": value_node.get("confidence", 0),
        }
        candidates.append(candidate)

    return candidates


# Known printed form headers/product names that should NOT be treated as field values
_FORM_HEADER_NOISE = {
    "amal women account", "account opening application", "account opening form",
    "bank al habib", "allied bank", "habib bank", "meezan bank", "standard chartered",
    "national bank", "ubi", "mcb", "ubl", "hbl", "bank alfalah",
    "for bank use only", "office use only", "branch use only",
}


def _is_noise_value(value: str) -> bool:
    """Return True if value appears to be a form header/noise rather than a user-filled value."""
    v = value.lower().strip()
    if not v:
        return True
    # Exact match against known noise strings
    if v in _FORM_HEADER_NOISE:
        return True
    # Partial match for long form names at start of value
    for noise in _FORM_HEADER_NOISE:
        if v.startswith(noise):
            return True
    return False


def merge_vlm_fields(
    graph_candidates: List[Dict[str, Any]],
    vlm_fields: List[Dict[str, Any]],
    ocr_blocks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Merge VLM-extracted fields with graph-based candidates.

    Strategy:
    - VLM is authoritative. If VLM returned >= 3 valid fields, graph-only candidates
      are suppressed unless they match a VLM field name (VLM acts as a filter).
    - If VLM returned < 3 fields (e.g., VLM failed or found little), graph candidates
      are included as-is for fallback.
    - VLM fields that match graph candidates are used to enrich (multi-source).
    - VLM-only fields are added directly.
    - If ocr_blocks provided, VLM-only field values are cross-checked against raw OCR
      text. Matches upgrade the field from VLM-only to multi-source for higher confidence.
    """
    # Filter out noisy vlm fields first
    clean_vlm_fields = [
        f for f in vlm_fields
        if not f.get("field_name", "").startswith("__")
        and not _is_noise_value(str(f.get("value", "")))
        and normalize_field_name(f.get("field_name", "")) not in ("", "unknown_field")
    ]

    vlm_is_authoritative = len(clean_vlm_fields) >= 3
    vlm_field_names = {normalize_field_name(f.get("field_name", "")) for f in clean_vlm_fields}

    merged: List[Dict[str, Any]] = []

    # Step 1: Include graph candidates, but only those VLM confirms (if VLM is authoritative)
    for candidate in graph_candidates:
        cname = candidate["field_name_normalized"]
        if vlm_is_authoritative and cname not in vlm_field_names:
            # Suppress graph-only noise when VLM is authoritative
            continue
        if _is_noise_value(candidate.get("value_normalized", "")):
            continue
        merged.append(candidate)

    # Step 2: Enrich graph candidates with VLM data, or add VLM-only candidates
    for vlm_field in clean_vlm_fields:
        vlm_name = normalize_field_name(vlm_field.get("field_name", ""))
        vlm_value = str(vlm_field.get("value", "")).strip()

        # Try to match with existing graph candidate
        matched = False
        for candidate in merged:
            if candidate["field_name_normalized"] == vlm_name:
                # Enrich existing candidate with VLM data
                if "qwen_vl" not in candidate.get("source_models", []):
                    candidate["source_models"].append("qwen_vl")
                # Prefer VLM value over potentially-noisy graph OCR value
                candidate["vlm_value"] = vlm_value
                candidate["vlm_confidence"] = vlm_field.get("confidence", 0.5)
                # Update the displayed value to VLM's (more reliable)
                candidate["value_normalized"] = vlm_value
                matched = True
                break

        if not matched:
            # Add as new candidate from VLM only
            field_type = infer_field_type(vlm_name, vlm_value)
            
            # Cross-check value against raw OCR blocks (corroboration)
            ocr_corroborated = False
            corroborating_engine = None
            if ocr_blocks and vlm_value:
                for engine_name, blocks in (ocr_blocks.items() if isinstance(ocr_blocks, dict) else [("ocr", ocr_blocks)]):
                    if _ocr_contains_value(blocks, vlm_value):
                        ocr_corroborated = True
                        corroborating_engine = engine_name
                        break
            
            source_models = ["qwen_vl"]
            if ocr_corroborated and corroborating_engine:
                source_models.append(corroborating_engine)
                logger.info(
                    "OCR corroborated VLM field '%s'='%s' via %s",
                    vlm_name, vlm_value, corroborating_engine
                )
            
            vlm_conf = vlm_field.get("confidence", 0.5)
            new_candidate = {
                "field_name_raw": vlm_field.get("field_name", vlm_name),
                "field_name_normalized": vlm_name,
                "value_raw": vlm_value,
                "value_normalized": vlm_value,
                "field_type": field_type,
                "label_bbox": None,
                "value_bbox": None,
                "geometry_score": 0.3 if ocr_corroborated else 0.0,  # Small geometry credit for corroborated fields
                "source_models": source_models,
                "label_confidence": 0,
                "value_confidence": 0.75 if ocr_corroborated else vlm_conf,  # Higher base confidence when OCR agrees
                "vlm_value": vlm_value,
                "vlm_confidence": vlm_conf,
                "vlm_only": not ocr_corroborated,
                "ocr_corroborated": ocr_corroborated,
            }
            merged.append(new_candidate)

    return merged
