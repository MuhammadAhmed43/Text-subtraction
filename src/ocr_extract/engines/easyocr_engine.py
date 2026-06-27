"""EasyOCR engine wrapper."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Global reader cache
_READER: Any = None

def _get_reader(lang: str) -> Any:
    """Initialize and return the EasyOCR Reader."""
    global _READER
    if _READER is not None:
        return _READER
        
    try:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        _READER = easyocr.Reader([lang], gpu=use_gpu, verbose=False)
        return _READER
    except ImportError:
        logger.error("EasyOCR is not installed. Install it with: pip install easyocr")
        return None

def run_easyocr(image_path: str, lang: str = "en") -> List[Dict[str, Any]]:
    """Run EasyOCR detection and recognition on an image."""
    reader = _get_reader(lang)
    if not reader:
        return []

    try:
        # returns a list of tuples: (bbox, text, prob)
        results = reader.readtext(image_path, detail=1, paragraph=False)
    except Exception:
        logger.exception("EasyOCR failed to process image: %s", image_path)
        return []

    blocks: List[Dict[str, Any]] = []
    
    for bbox, text, prob in results:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        blocks.append({
            "text": str(text).strip(),
            "bbox": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            },
            "confidence": float(prob),
            "model": "easyocr"
        })

    return blocks
