"""Preprocessor - creates image variants for OCR (deskew, denoise, contrast, binarize)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _to_cv2(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an OpenCV BGR array."""
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return arr


def _to_pil(arr: np.ndarray) -> Image.Image:
    """Convert an OpenCV array to a PIL Image."""
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def make_gray(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def make_contrast(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    gray = make_gray(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def make_binarized(img: np.ndarray) -> np.ndarray:
    """Adaptive thresholding for clear text/background separation."""
    gray = make_gray(img)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )


def make_denoised(img: np.ndarray) -> np.ndarray:
    """Light denoise without destroying handwriting strokes."""
    gray = make_gray(img)
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def deskew_image(img: np.ndarray) -> np.ndarray:
    """Deskew using Hough line detection."""
    gray = make_gray(img)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return img

    median_angle = np.median(angles)
    if abs(median_angle) < 0.3:
        return img  # Already straight enough

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    if img.ndim == 3:
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    logger.debug("Deskewed by %.2f degrees", median_angle)
    return rotated


def create_variants(
    page_image_path: str,
    output_dir: Path,
    config: PipelineConfig,
) -> Dict[str, str]:
    """Create all image variants for a single page."""
    output_dir.mkdir(parents=True, exist_ok=True)
    variants: Dict[str, str] = {}

    img_pil = Image.open(page_image_path)
    img_cv = _to_cv2(img_pil)

    # Optional deskew first
    if config.preprocess.enable_deskew:
        img_cv = deskew_image(img_cv)

    stem = Path(page_image_path).stem

    for variant_name in config.preprocess.variants:
        try:
            if variant_name == "original":
                out = img_cv
            elif variant_name == "gray":
                out = make_gray(img_cv)
            elif variant_name == "contrast":
                out = make_contrast(img_cv)
            elif variant_name == "binarized":
                out = make_binarized(img_cv)
            elif variant_name == "denoised":
                out = make_denoised(img_cv)
            else:
                logger.warning("Unknown variant: %s", variant_name)
                continue

            out_path = output_dir / f"{stem}_{variant_name}.png"
            if out.ndim == 2:
                Image.fromarray(out, mode="L").save(str(out_path))
            else:
                _to_pil(out).save(str(out_path))

            variants[variant_name] = str(out_path)
        except Exception as e:
            logger.error("Failed to create variant %s: %s", variant_name, e)

    return variants


def create_all_variants(
    manifest: dict, config: PipelineConfig
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Create variants for all pages in the manifest.
    
    Returns: {doc_id: {page_id: {variant_name: path}}}
    """
    work_dir = config.work_path or Path("work")
    variants_dir = work_dir / "variants"
    all_variants: Dict[str, Dict[str, Dict[str, str]]] = {}

    for doc in manifest.get("documents", []):
        doc_id = doc["document_id"]
        all_variants[doc_id] = {}
        for page in doc.get("pages", []):
            page_id = page["page_id"]
            page_dir = variants_dir / doc_id / page_id
            variants = create_variants(page["image_path"], page_dir, config)
            # Always include original page path
            variants["original_page"] = page["image_path"]
            all_variants[doc_id][page_id] = variants
            logger.debug("Created %d variants for %s", len(variants), page_id)

    return all_variants
