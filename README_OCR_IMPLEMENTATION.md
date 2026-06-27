# Open Source OCR/VLM Document Extraction Implementation

This documentation set describes a zero-dollar, local-first implementation for extracting structured data from handwritten and mixed printed forms. It is designed for the PDFs in this workspace and for future documents whose structure is unknown.

## Goal

Build an automated document extraction pipeline that:

- Uses open-source code and open-weight models only.
- Runs locally or on free infrastructure.
- Handles scanned PDFs, photos, printed text, handwriting, tables, checkboxes, signatures, and stamps.
- Avoids manual validation during normal processing.
- Never silently accepts weak results: every extracted field carries evidence and confidence.
- Works without fixed templates, while still allowing optional schemas when a business process needs them.

## Current Workspace Inputs

The workspace currently contains these PDF files:

- `AOF_01.pdf`
- `AOF02.pdf`
- `AOF_03.pdf`
- `AOF_04.pdf`
- `AOF_04 (1).pdf`
- `AOF_05.pdf`
- `AOF_06.pdf`
- `AOF_06 (1).pdf`
- `AOF_07.pdf`

## Documentation Map

- `OCR_IMPLEMENTATION_PLAN.md`: phased build plan and deliverables.
- `OCR_ARCHITECTURE.md`: end-to-end system design.
- `OCR_SETUP.md`: local setup, dependencies, hardware profiles, and folder structure.
- `OCR_PIPELINE.md`: exact processing flow from PDF to JSON/CSV.
- `OCR_EXTRACTION_SCHEMA.md`: output contracts and JSON schemas.
- `OCR_VALIDATION.md`: automated confidence scoring and no-manual-validation rules.
- `OCR_MODEL_EVALUATION.md`: model choices, licenses, benchmark process, and fallbacks.
- `OCR_RUNBOOK.md`: day-to-day commands, monitoring, troubleshooting, and release checklist.

## Recommended Default Stack

Use this combination first:

- PaddleOCR for layout-aware OCR and structured text.
- Tesseract as a fast baseline and printed-label cross-check.
- TrOCR handwritten for cropped handwritten text lines.
- Qwen3-VL-2B-Instruct for schema-free visual extraction and reasoning.
- Docling for document conversion and additional PDF structure extraction.

Optional additions:

- olmOCR when an NVIDIA GPU with enough VRAM is available.
- Surya only when its model-weight license fits the use case.

## Primary Outputs

Each document should produce:

- `raw_text.md`: all readable text in natural order.
- `layout.json`: text blocks, bounding boxes, model outputs, and confidence.
- `extracted_fields.json`: final key-value extraction with evidence.
- `extracted_fields.csv`: flat export for spreadsheet/database ingestion.
- `confidence_report.json`: agreement, uncertainty, and failure reasons.
- `evidence/`: cropped image snippets for every accepted field.

## Implementation Principle

The system should prefer evidence and agreement over a single model response. A field is accepted only when OCR, VLM, validators, and layout evidence support it strongly enough.

