"""CLI entry point - python -m ocr_extract run --input <path> --output <path>."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Set these before ANY paddle or paddleocr imports happen anywhere in the pipeline
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
# Force pure-Python protobuf to avoid crash with protobuf>=4.x (paddle needs <=3.20)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, TimeElapsedColumn

from . import __version__
from .config import PipelineConfig, load_config
from .input_manager import build_manifest, discover_inputs
from .preprocessor import create_all_variants
from .layout_graph import build_layout_graph
from .extractor import extract_fields_from_graph, merge_vlm_fields
from .validator import validate_candidates
from .exporter import export_all

console = Console()
logger = logging.getLogger("ocr_extract")


def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    """Configure logging with Rich handler and file output."""
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO

    # Rich console handler
    rich_handler = RichHandler(
        console=console, show_time=True, show_path=False,
        markup=True, rich_tracebacks=True,
    )
    rich_handler.setLevel(level)

    # File handler
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    ))

    root = logging.getLogger("ocr_extract")
    root.setLevel(logging.DEBUG)
    root.addHandler(rich_handler)
    root.addHandler(file_handler)


def run_ocr_engines(
    page_image_path: str,
    variant_paths: Dict[str, str],
    config: PipelineConfig,
) -> Dict[str, List[Dict]]:
    """Run all configured OCR engines on a page and its variants.
    
    Returns: {engine_name: [blocks]}
    """
    results: Dict[str, List[Dict]] = {}

    # Choose best variant for each engine
    contrast_path = variant_paths.get("contrast", page_image_path)
    original_path = variant_paths.get("original", page_image_path)

    enabled = config.ocr.engines

    # PaddleOCR
    if "paddleocr" in enabled:
        try:
            from .engines.paddleocr_engine import run_paddleocr
            t0 = time.time()
            blocks = run_paddleocr(contrast_path, lang=config.ocr.paddle_lang)
            elapsed = time.time() - t0
            results["paddleocr"] = blocks
            logger.info("PaddleOCR: %d blocks in %.1fs", len(blocks), elapsed)
        except Exception as e:
            logger.warning("PaddleOCR failed: %s", e)
            results["paddleocr"] = []

    # Tesseract
    if "tesseract" in enabled:
        try:
            from .engines.tesseract_engine import run_tesseract, check_tesseract_available
            if check_tesseract_available():
                t0 = time.time()
                blocks = run_tesseract(contrast_path, lang=config.ocr.tesseract_lang)
                elapsed = time.time() - t0
                results["tesseract"] = blocks
                logger.info("Tesseract: %d blocks in %.1fs", len(blocks), elapsed)
            else:
                logger.warning("Tesseract not installed, skipping")
                results["tesseract"] = []
        except Exception as e:
            logger.warning("Tesseract failed: %s", e)
            results["tesseract"] = []

    return results


def run_vlm_extraction_step(
    page_image_path: str,
    config: PipelineConfig,
    page_number: int,
) -> List[Dict[str, Any]]:
    """Run VLM extraction on a page image."""
    if "qwen_vl" not in config.ocr.engines:
        return []

    try:
        from .engines.vlm_engine import run_vlm_extraction, check_ollama_available
        if not check_ollama_available():
            logger.warning("Ollama not available, skipping VLM extraction")
            return []

        t0 = time.time()
        fields = run_vlm_extraction(
            page_image_path,
            prompt=config.vlm_prompt,
            model=config.ocr.vlm_model,
            page_number=page_number,
        )
        elapsed = time.time() - t0
        logger.info("VLM: %d fields in %.1fs", len(fields), elapsed)
        return fields
    except Exception as e:
        logger.warning("VLM extraction failed: %s", e)
        return []


def run_trocr_on_handwritten_regions(
    layout_nodes: List[Dict],
    page_image_path: str,
    config: PipelineConfig,
) -> List[Dict]:
    """Detect handwritten regions and run TrOCR on crops."""
    if "trocr_handwritten" not in config.ocr.engines:
        return []

    # Find value nodes that might be handwritten
    # (low confidence, or classified as "text"/"value")
    import cv2
    from .engines.trocr_engine import run_trocr_on_crop

    img = cv2.imread(page_image_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    results = []

    for node in layout_nodes:
        if node["node_type"] not in ("value", "text"):
            continue
        if node.get("confidence", 1.0) > 0.85:
            continue  # Already confident, skip

        bbox = node["bbox"]
        padding = 12
        x1 = max(0, bbox["x1"] - padding)
        y1 = max(0, bbox["y1"] - padding)
        x2 = min(w, bbox["x2"] + padding)
        y2 = min(h, bbox["y2"] + padding)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Save temp crop
        work_dir = config.work_path or Path("work")
        crops_dir = work_dir / "crops" / "temp"
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crops_dir / f"crop_{node['node_id']}.png"
        cv2.imwrite(str(crop_path), crop)

        try:
            result = run_trocr_on_crop(str(crop_path), model_name=config.ocr.trocr_model)
            if result.get("text"):
                result["node_id"] = node["node_id"]
                result["bbox"] = bbox
                results.append(result)
        except Exception as e:
            logger.debug("TrOCR failed on crop %s: %s", node["node_id"], e)

    if results:
        logger.info("TrOCR: processed %d handwritten crops", len(results))
    return results


def process_document(
    doc_entry: Dict[str, Any],
    config: PipelineConfig,
    all_variants: Dict[str, Dict[str, str]],
) -> None:
    """Process a single document through the full pipeline."""
    doc_id = doc_entry["document_id"]
    source_file = doc_entry["source_file"]
    output_dir = (config.output_path or Path("data/output")) / doc_id

    logger.info("-" * 60)
    logger.info("Processing: %s (doc_id=%s)", source_file, doc_id)
    logger.info("-" * 60)

    all_layout_graphs = []
    all_candidates = []
    page_texts: Dict[int, str] = {}
    page_image_paths: Dict[int, str] = {}
    model_runtimes: Dict[str, float] = {}
    doc_start = time.time()

    for page_info in doc_entry.get("pages", []):
        page_id = page_info["page_id"]
        page_num = page_info["page_number"]
        page_image_path = page_info["image_path"]
        page_image_paths[page_num] = page_image_path

        logger.info("-- Page %d --", page_num)

        # Get variants for this page
        variants = all_variants.get(doc_id, {}).get(page_id, {})
        if not variants:
            variants = {"original": page_image_path}

        # Step 1: Run OCR engines
        t0 = time.time()
        ocr_results = run_ocr_engines(page_image_path, variants, config)
        model_runtimes["ocr_total"] = model_runtimes.get("ocr_total", 0) + (time.time() - t0)

        # Collect raw text from all engines
        all_text_lines = []
        for engine_name, blocks in ocr_results.items():
            model_runtimes[engine_name] = model_runtimes.get(engine_name, 0)
            for block in blocks:
                all_text_lines.append(block.get("text", ""))

        page_texts[page_num] = "\n".join(all_text_lines)

        # Step 2: Build layout graph
        layout_graph = build_layout_graph(ocr_results, page_id, doc_id)
        all_layout_graphs.append(layout_graph)

        # Step 3: Run TrOCR on handwritten regions
        trocr_results = run_trocr_on_handwritten_regions(
            layout_graph.get("nodes", []), page_image_path, config
        )

        # Enrich layout nodes with TrOCR results
        nodes_by_id = {n["node_id"]: n for n in layout_graph.get("nodes", [])}
        for tr in trocr_results:
            node = nodes_by_id.get(tr.get("node_id"))
            if node:
                node["trocr_text"] = tr["text"]
                node["trocr_confidence"] = tr.get("confidence", 0.5)
                if "trocr_handwritten" not in node.get("source_models", []):
                    node["source_models"].append("trocr_handwritten")

        # Step 4: Run VLM extraction
        t0 = time.time()
        vlm_fields = run_vlm_extraction_step(page_image_path, config, page_num)
        model_runtimes["qwen_vl"] = model_runtimes.get("qwen_vl", 0) + (time.time() - t0)

        if vlm_fields and not page_texts.get(page_num):
            visible_vlm_lines = [
                f"{field.get('field_name', 'unknown')}: {field.get('value', '')}"
                for field in vlm_fields
                if str(field.get("value", "")).strip()
            ]
            if visible_vlm_lines:
                page_texts[page_num] = "\n".join(visible_vlm_lines)

        # Step 5: Extract field candidates from graph
        graph_candidates = extract_fields_from_graph(layout_graph)

        # Step 6: Merge with VLM fields (pass raw OCR blocks for cross-corroboration)
        merged = merge_vlm_fields(graph_candidates, vlm_fields, ocr_blocks=ocr_results)

        # Add page number to all candidates
        for c in merged:
            c["page_number"] = page_num

        all_candidates.extend(merged)

    # Step 7: Validate all candidates
    validated = validate_candidates(all_candidates, config.validation)

    model_runtimes["total"] = time.time() - doc_start

    # Step 8: Export everything
    export_all(
        document_id=doc_id,
        source_file=source_file,
        validated_candidates=validated,
        layout_graphs=all_layout_graphs,
        page_texts=page_texts,
        model_runtimes=model_runtimes,
        output_dir=output_dir,
        page_image_paths=page_image_paths,
        save_evidence=config.outputs.save_evidence_crops,
    )

    # Summary
    accepted = sum(1 for c in validated if c["status"] == "accepted")
    uncertain = sum(1 for c in validated if c["status"] == "uncertain")
    rejected = sum(1 for c in validated if c["status"] == "rejected")
    missing = sum(1 for c in validated if c["status"] == "missing")
    
    console.print(
        f"\n[bold green]OK[/] {source_file}: "
        f"[green]{accepted} accepted[/], "
        f"[yellow]{uncertain} uncertain[/], "
        f"[red]{rejected} rejected[/], "
        f"[dim]{missing} missing[/] "
        f"({time.time() - doc_start:.1f}s)"
    )

    # Print extracted results to terminal
    from rich.table import Table
    table = Table(title=f"Extracted Fields - {source_file}", show_header=True, header_style="bold magenta")
    table.add_column("Field")
    table.add_column("Value")
    table.add_column("Status")
    table.add_column("Confidence", justify="right")

    for c in validated:
        status = c["status"]
        if status in ("accepted", "uncertain"):
            color = "green" if status == "accepted" else "yellow"
            table.add_row(
                str(c.get("field_name_raw", c.get("field_name_normalized", ""))),
                str(c.get("value_normalized", "")),
                f"[{color}]{status}[/]",
                f"{c.get('final_score', 0):.2f}"
            )
            
    if table.row_count > 0:
        console.print(table)
    else:
        console.print("[yellow]No valid fields extracted to display.[/]")

    # Copy CSV to a flattened top-level folder for easy user access
    csv_path = output_dir / "extracted_fields.csv"
    if csv_path.exists():
        top_csv_dir = (config.output_path or Path("data/output")).parent / "csv_exports"
        top_csv_dir.mkdir(parents=True, exist_ok=True)
        target_csv_name = f"{Path(source_file).stem}.csv"
        import shutil
        shutil.copy2(csv_path, top_csv_dir / target_csv_name)



def run_pipeline(args: argparse.Namespace) -> None:
    """Main pipeline execution."""
    if not args.input:
        from rich.prompt import Prompt
        args.input = Prompt.ask("[bold cyan]Please enter the input file or directory path[/]")

    # Resolve paths
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    project_root = Path(os.getcwd()).resolve()
    work_path = project_root / "work"
    log_dir = project_root / "logs"

    # Load config
    config_path = args.config if args.config else str(project_root / "configs" / "default.yaml")
    config = load_config(config_path)
    config.input_path = input_path
    config.output_path = output_path
    config.work_path = work_path
    config.project_root = project_root

    # Setup logging
    setup_logging(log_dir, verbose=args.verbose)

    console.print(f"\n[bold cyan]OCR Extraction Pipeline v{__version__}[/]")
    console.print(f"Input: {input_path}")
    console.print(f"Output: {output_path}")
    console.print(f"Config: {config_path}")
    console.print()

    # Step 1: Discover inputs
    input_files = discover_inputs(input_path)
    if not input_files:
        console.print("[bold red]No supported files found![/]")
        sys.exit(1)
    console.print(f"Found [bold]{len(input_files)}[/] document(s)")

    # Step 2: Build manifest (renders pages)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Rendering pages...", total=len(input_files))
        manifest = build_manifest(input_files, config)
        progress.update(task, completed=len(input_files))

    total_pages = sum(d["page_count"] for d in manifest["documents"])
    console.print(f"Rendered [bold]{total_pages}[/] page(s) across {len(manifest['documents'])} document(s)")

    # Step 3: Preprocess variants
    console.print("Creating image variants...")
    all_variants = create_all_variants(manifest, config)

    # Step 4: Process each document
    for doc_entry in manifest["documents"]:
        if doc_entry["status"] == "render_failed":
            console.print(f"[bold red]FAILED[/] {doc_entry['source_file']}: render failed, skipping")
            continue
        process_document(doc_entry, config, all_variants)

    console.print(f"\n[bold green]Pipeline complete![/] Outputs (including [bold]extracted_fields.csv[/]) are saved in: {output_path}")


def main() -> None:
    """CLI argument parser and dispatcher."""
    parser = argparse.ArgumentParser(
        prog="ocr_extract",
        description="OCR/VLM Document Extraction Pipeline",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Extract data from documents")
    run_parser.add_argument("--input", "-i", default=None, help="Input file or directory (if omitted, you will be prompted)")
    run_parser.add_argument("--output", "-o", default="data/output", help="Output directory")
    run_parser.add_argument("--config", "-c", default=None, help="Config YAML file")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
