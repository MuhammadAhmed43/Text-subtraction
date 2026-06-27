"""PaddleOCR engine wrapper for the OCR extraction pipeline.

Provides a unified interface to run PaddleOCR detection and recognition
on images, returning standardized OCR block dictionaries.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional

# Must be set BEFORE any paddle/paddleocr import to bypass protobuf>=4.x crash.
# paddle was built against protobuf<=3.20; newer versions break descriptor creation.
# Setting this env var forces the pure-Python protobuf implementation which is compatible.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

logger = logging.getLogger(__name__)

# Singleton reader cache so models are only loaded once per process
_OCR_INSTANCE: Any = None


def _configure_paddle_runtime() -> None:
    """Set conservative CPU/GPU runtime flags before importing Paddle."""
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_use_onednn"] = "0"
    os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
    # Suppress excessive paddle logging
    os.environ.setdefault("GLOG_minloglevel", "3")


def _polygon_to_bbox(points: List[List[float]]) -> Dict[str, int]:
    """Convert a polygon to an axis-aligned bounding box."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {
        "x1": int(min(xs)),
        "y1": int(min(ys)),
        "x2": int(max(xs)),
        "y2": int(max(ys)),
    }


def _coerce_polygon(points: Any) -> Optional[List[List[float]]]:
    """Convert Paddle polygon variants to a list of [x, y] points."""
    if points is None:
        return None
    if hasattr(points, "tolist"):
        points = points.tolist()

    try:
        polygon = [[float(p[0]), float(p[1])] for p in points]
    except (TypeError, ValueError, IndexError):
        return None

    return polygon if polygon else None


def _parse_result_dict(result: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Parse PaddleOCR v3/PaddleX-style dict outputs."""
    texts = result.get("rec_texts") or result.get("texts") or []
    scores = result.get("rec_scores") or result.get("scores") or []
    polygons = (
        result.get("rec_polys")
        or result.get("dt_polys")
        or result.get("rec_boxes")
        or result.get("boxes")
        or []
    )

    for idx, text in enumerate(texts):
        polygon = _coerce_polygon(polygons[idx] if idx < len(polygons) else None)
        if not polygon:
            continue

        confidence = scores[idx] if idx < len(scores) else 0.5
        yield {
            "text": str(text),
            "bbox": _polygon_to_bbox(polygon),
            "confidence": float(confidence),
            "model": "paddleocr",
        }


def _parse_ocr_output(ocr_output: Any) -> List[Dict[str, Any]]:
    """Normalize old and new PaddleOCR result formats."""
    results: List[Dict[str, Any]] = []

    if not ocr_output:
        return results

    for page in ocr_output:
        if page is None:
            continue

        if hasattr(page, "json"):
            try:
                page = page.json
            except Exception:
                pass

        if isinstance(page, dict):
            results.extend(_parse_result_dict(page))
            continue

        for line in page:
            try:
                polygon_points = _coerce_polygon(line[0])
                if not polygon_points:
                    continue
                text, confidence = line[1]
                results.append(
                    {
                        "text": str(text),
                        "bbox": _polygon_to_bbox(polygon_points),
                        "confidence": float(confidence),
                        "model": "paddleocr",
                    }
                )
            except (IndexError, TypeError, ValueError) as exc:
                logger.debug("Failed to parse PaddleOCR line %r: %s", line, exc)
                continue

    return results


def _create_paddleocr(lang: str) -> Any:
    """Create RapidOCR (which runs PaddleOCR models via ONNX Runtime)."""
    from rapidocr_onnxruntime import RapidOCR
    # rapidocr uses 'ch' for chinese, 'en' for english
    return RapidOCR()


def run_paddleocr(image_path: str, lang: str = "en") -> List[Dict[str, Any]]:
    """
    Run RapidOCR (ONNX port of PaddleOCR) on an image and return extracted text blocks.
    Uses a process-level singleton so models are loaded only once.
    """
    global _OCR_INSTANCE
    
    if _OCR_INSTANCE is None:
        _OCR_INSTANCE = _create_paddleocr(lang)

    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # result format from rapidocr: list of [dt_boxes, rec_res, score]
    # where dt_boxes is 4 points, rec_res is text, score is float
    result, _ = _OCR_INSTANCE(img)
    blocks = []
    
    if result is None:
        return blocks

    for line in result:
        # line = [box, text, score]
        box = line[0]
        text = line[1]
        conf = float(line[2])

        # rapidocr box is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        try:
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            blocks.append({
                "text": text,
                "confidence": conf,
                "bbox": {"x1": int(x_min), "y1": int(y_min), "x2": int(x_max), "y2": int(y_max)},
                "model": "paddleocr" # Keep engine name same for downstream
            })
        except Exception as e:
            logger.warning("Failed to parse RapidOCR box: %s", e)

    return blocks
