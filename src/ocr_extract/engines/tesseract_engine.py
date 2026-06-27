"""Tesseract OCR engine wrapper."""

from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _find_tesseract_cmd() -> str:
    """Find Tesseract on PATH, env vars, or common Windows install paths."""
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        return env_cmd

    path_cmd = shutil.which("tesseract")
    if path_cmd:
        return path_cmd

    common_paths = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Tesseract-OCR" / "tesseract.exe",
        Path(r"C:\Tesseract-OCR\tesseract.exe"),
    ]
    for candidate in common_paths:
        if candidate.exists():
            return str(candidate)
    return ""


def check_tesseract_available() -> bool:
    """Check whether the Tesseract binary is available."""
    try:
        import pytesseract

        configured_cmd = _find_tesseract_cmd()
        if configured_cmd:
            pytesseract.pytesseract.tesseract_cmd = configured_cmd

        pytesseract.get_tesseract_version()
        logger.debug("Tesseract binary found.")
        return True
    except Exception:
        found = bool(_find_tesseract_cmd())
        if found:
            logger.debug("Tesseract binary found by path lookup.")
        else:
            logger.warning("Tesseract binary not found.")
        return found


def run_tesseract(image_path: str, lang: str = "eng") -> List[Dict]:
    """Run Tesseract OCR on an image and return line-level OCR blocks."""
    try:
        import pytesseract
        from pytesseract import Output

        configured_cmd = _find_tesseract_cmd()
        if configured_cmd:
            pytesseract.pytesseract.tesseract_cmd = configured_cmd
    except ImportError:
        logger.error("pytesseract is not installed. Install it with: pip install pytesseract")
        return []

    try:
        data: Dict = pytesseract.image_to_data(
            image_path,
            lang=lang,
            output_type=Output.DICT,
        )
    except Exception:
        logger.exception("Tesseract failed to process image: %s", image_path)
        return []

    line_words: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    n_entries = len(data.get("text", []))

    for idx in range(n_entries):
        text = str(data["text"][idx]).strip()
        try:
            confidence = float(data["conf"][idx])
        except (TypeError, ValueError):
            confidence = -1.0

        if not text or confidence < 10:
            continue

        key = (
            int(data["block_num"][idx]),
            int(data["par_num"][idx]),
            int(data["line_num"][idx]),
        )
        line_words[key].append(idx)

    results: List[Dict] = []

    for key in sorted(line_words.keys()):
        indices = line_words[key]

        words: List[str] = []
        confidences: List[float] = []
        x1_min = int(data["left"][indices[0]])
        y1_min = int(data["top"][indices[0]])
        x2_max = 0
        y2_max = 0

        for idx in indices:
            words.append(str(data["text"][idx]).strip())
            confidences.append(float(data["conf"][idx]))

            left = int(data["left"][idx])
            top = int(data["top"][idx])
            width = int(data["width"][idx])
            height = int(data["height"][idx])

            x1_min = min(x1_min, left)
            y1_min = min(y1_min, top)
            x2_max = max(x2_max, left + width)
            y2_max = max(y2_max, top + height)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        results.append(
            {
                "text": " ".join(words),
                "bbox": {
                    "x1": x1_min,
                    "y1": y1_min,
                    "x2": x2_max,
                    "y2": y2_max,
                },
                "confidence": round(avg_conf / 100.0, 4),
                "model": "tesseract",
            }
        )

    logger.info("Tesseract extracted %d line block(s) from %s", len(results), image_path)
    return results
