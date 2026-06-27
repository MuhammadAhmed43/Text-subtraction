"""VLM Engine - Qwen2.5-VL via Ollama for schema-free visual extraction."""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Module-level cache
_ollama_available: Optional[bool] = None
_resolved_model: Optional[str] = None  # cache for auto-detected model


def check_ollama_available() -> bool:
    """Check if Ollama Python client is installed and server reachable."""
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available
    try:
        import ollama
        ollama.list()
        _ollama_available = True
    except Exception:
        _ollama_available = False
    return _ollama_available


def _resolve_vlm_model(requested_model: str) -> Optional[str]:
    """Resolve the best available VLM model.
    
    If the requested model is available, use it.
    Otherwise, fall back to any qwen-vl variant that is installed.
    Returns None if no suitable model is found.
    """
    global _resolved_model
    if _resolved_model is not None:
        return _resolved_model
    try:
        import ollama
        model_list = ollama.list()
        # Handle both old (dict) and new (object) ollama API responses
        if hasattr(model_list, 'models'):
            available = [m.model for m in model_list.models]
        else:
            available = [m.get('name', m.get('model', '')) for m in model_list.get('models', [])]

        # Check if exact model is available
        if requested_model in available or any(requested_model in a for a in available):
            _resolved_model = requested_model
            return _resolved_model

        # Fall back: prefer qwen2.5vl, then qwen2-vl, then any vl model
        for prefix in ("qwen2.5vl", "qwen2-vl", "qwen", "llava", "moondream"):
            for a in available:
                if prefix in a.lower():
                    logger.warning(
                        "Requested model '%s' not found. Falling back to '%s'",
                        requested_model, a
                    )
                    _resolved_model = a
                    return _resolved_model

        logger.warning("No VLM model found in Ollama. Available: %s", available)
        return None
    except Exception as e:
        logger.warning("Could not list Ollama models: %s", e)
        return requested_model  # try anyway


def _image_to_base64(image_path: str) -> str:
    """Read image file and return base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_json_from_response(text: str) -> Any:
    """Parse JSON from VLM response, handling markdown code fences."""
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"\[.*\]",
        r"\{.*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if "```" in pattern else match.group(0))
            except (json.JSONDecodeError, IndexError):
                continue

    logger.warning("Could not parse JSON from VLM response")
    return None


def normalize_field_name(raw_name: str) -> str:
    """Normalize field name inline to avoid circular imports."""
    name = raw_name.strip().rstrip(":").strip()
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", "_", name.lower()).strip("_")


def is_checkbox_value(val: str, name: str) -> bool:
    """Detect if a value likely contains multiple checkbox options instead of a single selection."""
    val = val.strip()
    if not val:
        return False
    # Common words in checkboxes/choices
    options_keywords = {
        "yes", "no", "present", "absent", "checked", "unchecked", 
        "classic", "gold", "platinum", "world", "paypak", "visa", "mastercard", "unionpay", 
        "mobilink", "telenor", "ufone", "zong", "english", "urdu", 
        "salaried", "pensioner", "student", "housewife", "unemployed", "business"
    }
    val_lower = val.lower()
    
    # Bracket indicators
    if any(box in val for box in ("[]", "[ ]", "[x]", "[X]", "( )", "()", "☑", "☒", "☐")):
        return True
        
    words = val.split()
    if len(words) >= 2:
        # Check if we see multiple choice keywords
        matches = sum(1 for w in words if w.lower().strip(",()[]") in options_keywords)
        if matches >= 2:
            return True
            
    # Or if the name suggests a checkbox group and we have multiple words
    name_lower = name.lower()
    if any(k in name_lower for k in ("scheme", "network", "type", "leaves", "language", "occupation", "status", "gender", "category")):
        if len(words) >= 2:
            return True
            
    return False


def run_vlm_extraction(
    image_path: str,
    prompt: str,
    model: str = "qwen2.5vl:3b",
    page_number: int = 1,
) -> List[Dict[str, Any]]:
    """Run VLM extraction on a single page image.
    
    Args:
        image_path: Path to the page image
        prompt: The extraction prompt
        model: Ollama model name
        page_number: Page number for metadata
        
    Returns:
        List of extracted field dicts with keys:
        field_name, value, page, confidence, uncertainty_reason, model
    """
    if not check_ollama_available():
        logger.warning("Ollama not available, skipping VLM extraction")
        return []

    try:
        import ollama

        # Auto-resolve the model (fall back to whatever is available)
        resolved = _resolve_vlm_model(model)
        if resolved is None:
            logger.warning("No VLM model available in Ollama, skipping VLM extraction")
            return []
        if resolved != model:
            model = resolved

        logger.info("Running VLM extraction on %s (model=%s)", Path(image_path).name, model)

        # Read image as bytes for Ollama
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_bytes],
            }],
            options={
                "temperature": 0.1,  # Low temperature for factual extraction
                "num_predict": 4096,
                "num_ctx": 8192,
            },
        )

        raw_text = response.get("message", {}).get("content", "")
        logger.debug("VLM raw response length: %d chars", len(raw_text))

        parsed = _extract_json_from_response(raw_text)
        if parsed is None:
            logger.error("VLM raw response was:\n%s", raw_text)
            logger.warning("VLM returned unparseable response for %s", image_path)
            return [{
                "field_name": "__raw_vlm_text__",
                "value": raw_text,
                "page": page_number,
                "confidence": 0.3,
                "uncertainty_reason": "Could not parse structured JSON from VLM",
                "model": "qwen_vl",
            }]

        # Normalize to list
        if isinstance(parsed, dict):
            parsed = [parsed]

        # Enrich each field
        results = []
        for field_data in parsed:
            if not isinstance(field_data, dict):
                continue
            field = {
                "field_name": str(field_data.get("field_name", "unknown")),
                "value": str(field_data.get("value", "")),
                "page": page_number,
                "confidence": float(field_data.get("confidence", 0.5)),
                "uncertainty_reason": field_data.get("uncertainty_reason"),
                "model": "qwen_vl",
            }
            results.append(field)

        # --- Phase 2: Targeted Checkbox Resolution ---
        checkbox_fields = []
        for field in results:
            val = field["value"]
            name = field["field_name"]
            if is_checkbox_value(val, name):
                checkbox_fields.append((name, val))

        if checkbox_fields:
            field_names_list = [item[0] for item in checkbox_fields]
            logger.info("Detecting checkbox fields for refinement: %s", field_names_list)
            
            checkbox_prompt = f"""Look closely at the checkboxes/options for these specific fields in the form: {field_names_list}.
For each field, inspect the image to find which option has a tick (✓), cross (X), or is filled/marked.
Return a STRICT JSON object mapping each field name to ONLY the single checked option text.
Example format:
{{
  "ATM/Debit Card Scheme": "Paypak",
  "Card Type": "Classic"
}}
Do not return any other text, markdown, or explanations. Only the JSON object.
"""
            try:
                checkbox_response = ollama.chat(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": checkbox_prompt,
                        "images": [image_bytes],
                    }],
                    options={
                        "temperature": 0.1,
                        "num_predict": 1024,
                        "num_ctx": 8192,
                    },
                )
                cb_text = checkbox_response.get("message", {}).get("content", "").strip()
                resolved = _extract_json_from_response(cb_text)
                
                if isinstance(resolved, dict):
                    # Helper for fuzzy matching resolved names back to fields
                    def get_resolved_val(f_name: str) -> Optional[str]:
                        f_norm = normalize_field_name(f_name)
                        for r_key, r_val in resolved.items():
                            r_norm = normalize_field_name(r_key)
                            if r_norm == f_norm or r_norm in f_norm or f_norm in r_norm:
                                return str(r_val).strip()
                        return None

                    for field in results:
                        name = field["field_name"]
                        resolved_val = get_resolved_val(name)
                        if resolved_val:
                            logger.info("Refined checkbox field: %s -> %s", name, resolved_val)
                            field["value"] = resolved_val
                            field["confidence"] = max(field["confidence"], 0.85)  # boost confidence since verified
            except Exception as cb_err:
                logger.warning("Targeted checkbox resolution failed: %s", cb_err)

        logger.info("VLM extracted %d fields from %s", len(results), Path(image_path).name)
        return results

    except Exception as e:
        logger.error("VLM extraction failed for %s: %s", image_path, e)
        return []


def run_vlm_on_crop(
    crop_path: str,
    model: str = "qwen2.5vl:3b",
) -> Dict[str, Any]:
    """Run VLM on a single crop to read handwritten text.
    
    Returns a dict with text, confidence, model.
    """
    if not check_ollama_available():
        return {"text": "", "confidence": 0.0, "model": "qwen_vl"}

    try:
        import ollama

        with open(crop_path, "rb") as f:
            image_bytes = f.read()

        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": "Read the handwritten text in this image. Return ONLY the text, nothing else.",
                "images": [image_bytes],
            }],
            options={"temperature": 0.1, "num_predict": 256},
        )

        text = response.get("message", {}).get("content", "").strip()
        return {"text": text, "confidence": 0.6, "model": "qwen_vl"}

    except Exception as e:
        logger.error("VLM crop read failed: %s", e)
        return {"text": "", "confidence": 0.0, "model": "qwen_vl"}
