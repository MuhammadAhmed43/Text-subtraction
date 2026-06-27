"""Exporter - writes JSON, CSV, Markdown, and evidence crops per document."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def save_evidence_crop(
    page_image_path: str,
    bbox: Dict[str, int],
    output_path: str,
    padding: int = 12,
) -> bool:
    """Crop and save evidence image for a field."""
    try:
        img = cv2.imread(page_image_path)
        if img is None:
            return False

        h, w = img.shape[:2]
        x1 = max(0, bbox["x1"] - padding)
        y1 = max(0, bbox["y1"] - padding)
        x2 = min(w, bbox["x2"] + padding)
        y2 = min(h, bbox["y2"] + padding)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, crop)
        return True
    except Exception as e:
        logger.error("Failed to save evidence crop: %s", e)
        return False


def export_raw_text(
    all_page_texts: Dict[int, str],
    source_file: str,
    output_path: Path,
) -> None:
    """Export raw_text.md with all text organized by page."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {source_file}\n"]

    for page_num in sorted(all_page_texts.keys()):
        text = all_page_texts[page_num]
        lines.append(f"\n## Page {page_num}\n")
        lines.append(text)
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Saved raw text to %s", output_path)


def export_layout_json(
    layout_graphs: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Save layout.json with all page layout graphs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(layout_graphs, f, indent=2, ensure_ascii=False)
    logger.info("Saved layout to %s", output_path)


def export_extracted_fields(
    document_id: str,
    source_file: str,
    validated_candidates: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Save extracted_fields.json per OCR_EXTRACTION_SCHEMA.md format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Separate accepted/uncertain from rejected
    fields = []
    unmapped = []
    for c in validated_candidates:
        field_entry = {
            "field_name": c.get("field_name_normalized", "unknown"),
            "display_name": c.get("field_name_raw", ""),
            "value": c.get("value_normalized", ""),
            "raw_value": c.get("value_raw", ""),
            "field_type": c.get("field_type", "text"),
            "status": c.get("status", "uncertain"),
            "confidence": c.get("final_score", 0),
            "page_number": c.get("page_number", 1),
            "evidence_crop": c.get("evidence_crop_path", ""),
            "source_models": c.get("source_models", []),
            "uncertainty_reason": c.get("uncertainty_reason"),
        }
        if c.get("status") in ("accepted", "uncertain"):
            fields.append(field_entry)
        else:
            unmapped.append(field_entry)

    # Overall stats
    overall_scores = [c.get("final_score", 0) for c in validated_candidates if c.get("final_score", 0) > 0]
    overall_confidence = sum(overall_scores) / len(overall_scores) if overall_scores else 0

    statuses = [c.get("status") for c in validated_candidates]
    if not statuses:
        overall_status = "no_fields_found"
    elif all(s == "accepted" for s in statuses):
        overall_status = "fully_accepted"
    elif any(s == "conflict" for s in statuses):
        overall_status = "has_conflicts"
    elif any(s == "rejected" for s in statuses):
        overall_status = "has_rejections"
    elif any(s == "uncertain" for s in statuses):
        overall_status = "accepted_with_uncertainties"
    elif any(s == "missing" for s in statuses):
        overall_status = "missing_fields_only"
    else:
        overall_status = "processed"

    doc = {
        "document_id": document_id,
        "source_file": source_file,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "0.1.0",
        "overall_status": overall_status,
        "overall_confidence": round(overall_confidence, 3),
        "fields": fields,
        "unmapped_fields": unmapped,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    logger.info("Saved extracted fields to %s (%d fields, %d unmapped)", output_path, len(fields), len(unmapped))


def export_csv(
    document_id: str,
    source_file: str,
    validated_candidates: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Save extracted_fields.csv per OCR_EXTRACTION_SCHEMA.md columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "document_id", "source_file", "field_name", "display_name",
        "value", "raw_value", "field_type", "status", "confidence",
        "page_number", "evidence_crop", "source_models", "uncertainty_reason",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for c in validated_candidates:
            row = {
                "document_id": document_id,
                "source_file": source_file,
                "field_name": c.get("field_name_normalized", ""),
                "display_name": c.get("field_name_raw", ""),
                "value": c.get("value_normalized", ""),
                "raw_value": c.get("value_raw", ""),
                "field_type": c.get("field_type", ""),
                "status": c.get("status", ""),
                "confidence": c.get("final_score", 0),
                "page_number": c.get("page_number", 1),
                "evidence_crop": c.get("evidence_crop_path", ""),
                "source_models": ",".join(c.get("source_models", [])),
                "uncertainty_reason": c.get("uncertainty_reason", ""),
            }
            writer.writerow(row)

    logger.info("Saved CSV to %s", output_path)


def export_confidence_report(
    document_id: str,
    validated_candidates: List[Dict[str, Any]],
    model_runtimes: Dict[str, float],
    output_path: Path,
) -> None:
    """Save confidence_report.json per OCR_EXTRACTION_SCHEMA.md."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    status_counts = {"accepted": 0, "uncertain": 0, "missing": 0, "conflict": 0, "rejected": 0}
    for c in validated_candidates:
        s = c.get("status", "rejected")
        if s in status_counts:
            status_counts[s] += 1

    warnings = []
    for c in validated_candidates:
        if c.get("status") in ("uncertain", "conflict"):
            warnings.append({
                "code": "LOW_CONFIDENCE" if c["status"] == "uncertain" else "FIELD_CONFLICT",
                "field_name": c.get("field_name_normalized", ""),
                "page_number": c.get("page_number", 1),
                "message": c.get("uncertainty_reason", ""),
            })

    report = {
        "document_id": document_id,
        "summary": status_counts,
        "model_runtime_seconds": model_runtimes,
        "retry_count": 0,
        "warnings": warnings,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved confidence report to %s", output_path)


def export_all(
    document_id: str,
    source_file: str,
    validated_candidates: List[Dict[str, Any]],
    layout_graphs: List[Dict[str, Any]],
    page_texts: Dict[int, str],
    model_runtimes: Dict[str, float],
    output_dir: Path,
    page_image_paths: Dict[int, str],
    save_evidence: bool = True,
) -> None:
    """Run all exports for a single document."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evidence crops if bboxes available
    if save_evidence:
        evidence_dir = output_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        for i, candidate in enumerate(validated_candidates):
            bbox = candidate.get("value_bbox")
            page_num = candidate.get("page_number", 1)
            page_img = page_image_paths.get(page_num)
            if bbox and page_img:
                field_name = candidate.get("field_name_normalized", f"field_{i}")
                crop_filename = f"page_{page_num:04d}_{field_name}_cand_{i:04d}.png"
                crop_path = evidence_dir / crop_filename
                if save_evidence_crop(page_img, bbox, str(crop_path)):
                    candidate["evidence_crop_path"] = str(crop_path)

    # Export all formats
    export_raw_text(page_texts, source_file, output_dir / "raw_text.md")
    export_layout_json(layout_graphs, output_dir / "layout.json")
    export_extracted_fields(document_id, source_file, validated_candidates, output_dir / "extracted_fields.json")
    export_csv(document_id, source_file, validated_candidates, output_dir / "extracted_fields.csv")
    export_confidence_report(document_id, validated_candidates, model_runtimes, output_dir / "confidence_report.json")

    logger.info("All exports complete for %s -> %s", source_file, output_dir)
