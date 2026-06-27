"""Input Manager - discovers files, computes hashes, renders PDF pages to images."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from .config import PipelineConfig

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def compute_file_hash(filepath: Path) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def make_document_id(filepath: Path, file_hash: str) -> str:
    """Create a stable document ID from filename + hash prefix."""
    stem = filepath.stem.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return f"{stem}_{file_hash[:8]}"


def discover_inputs(input_path: Path) -> List[Path]:
    """Find all supported files in the input path."""
    files: List[Path] = []
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(input_path)
    elif input_path.is_dir():
        for item in sorted(input_path.iterdir()):
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(item)
    return files


def render_pdf_pages(
    pdf_path: Path, output_dir: Path, dpi: int = 350, image_format: str = "png"
) -> List[Dict[str, Any]]:
    """Render all pages of a PDF to high-resolution images using PyMuPDF."""
    pages_info: List[Dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.error("Failed to open PDF %s: %s", pdf_path, e)
        return pages_info

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        page_filename = f"page_{page_num + 1:04d}.{image_format}"
        page_path = output_dir / page_filename
        pix.save(str(page_path))

        pages_info.append({
            "page_number": page_num + 1,
            "image_path": str(page_path),
            "width": pix.width,
            "height": pix.height,
            "dpi": dpi,
        })
        logger.debug("Rendered page %d of %s (%dx%d)", page_num + 1, pdf_path.name, pix.width, pix.height)

    doc.close()
    return pages_info


def render_image_input(
    image_path: Path, output_dir: Path
) -> List[Dict[str, Any]]:
    """Copy/normalize an image input as a single 'page'."""
    output_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path)

    # Normalize EXIF orientation
    from PIL import ImageOps
    img = ImageOps.exif_transpose(img)

    page_path = output_dir / f"page_0001.png"
    img.save(str(page_path), "PNG")

    return [{
        "page_number": 1,
        "image_path": str(page_path),
        "width": img.width,
        "height": img.height,
        "dpi": 0,  # unknown for image inputs
    }]


def build_manifest(
    input_files: List[Path], config: PipelineConfig
) -> Dict[str, Any]:
    """Build the document manifest and render all pages."""
    work_dir = config.work_path or Path("work")
    pages_dir = work_dir / "pages"
    manifests_dir = work_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "batch_id": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": [],
    }

    for filepath in input_files:
        file_hash = compute_file_hash(filepath)
        doc_id = make_document_id(filepath, file_hash)
        doc_pages_dir = pages_dir / doc_id

        logger.info("Processing input: %s -> doc_id=%s", filepath.name, doc_id)

        if filepath.suffix.lower() == ".pdf":
            pages = render_pdf_pages(
                filepath, doc_pages_dir, config.render.dpi, config.render.image_format
            )
        else:
            pages = render_image_input(filepath, doc_pages_dir)

        page_entries = []
        for p in pages:
            page_id = f"{doc_id}_p{p['page_number']:04d}"
            page_entries.append({
                "page_id": page_id,
                "page_number": p["page_number"],
                "image_path": p["image_path"],
                "width": p["width"],
                "height": p["height"],
                "dpi": p["dpi"],
            })

        doc_entry = {
            "document_id": doc_id,
            "source_file": str(filepath.name),
            "source_path": str(filepath),
            "source_hash": file_hash,
            "file_type": filepath.suffix.lower().lstrip("."),
            "page_count": len(page_entries),
            "status": "ready" if page_entries else "render_failed",
            "pages": page_entries,
        }
        manifest["documents"].append(doc_entry)

    # Save manifest
    manifest_path = manifests_dir / "document_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Manifest saved to %s", manifest_path)

    return manifest
