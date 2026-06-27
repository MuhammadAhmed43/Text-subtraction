# OCR/VLM Extraction Architecture

## High-Level Flow

```text
Input PDFs/images
  -> render pages
  -> preprocess variants
  -> OCR ensemble
  -> layout graph
  -> schema-free extraction
  -> confidence and validation
  -> JSON/CSV/Markdown/evidence exports
```

## Component Responsibilities

### 1. Input Manager

Responsibilities:

- Discover supported files.
- Create document IDs.
- Track source metadata.
- Render PDFs to page images.
- Avoid duplicate processing using hashes.

Inputs:

- PDF files.
- Image files.

Outputs:

- `manifest.json`
- page images
- source metadata

### 2. Preprocessor

Responsibilities:

- Rotate pages.
- Deskew pages.
- Normalize contrast.
- Denoise images.
- Generate multiple image variants.
- Preserve coordinate transforms.

Why this matters:

OCR models often disagree because they see different visual signals. Multiple variants let the pipeline choose the best result without manual tuning.

### 3. OCR Ensemble

Responsibilities:

- Run several independent OCR engines.
- Keep word and line bounding boxes.
- Keep confidence scores where available.
- Produce comparable text blocks.

Default engines:

- PaddleOCR for strong multilingual OCR and layout.
- Tesseract for baseline printed OCR and TSV/hOCR output.
- Docling for PDF structure and document conversion.
- TrOCR for cropped handwritten text lines.
- Qwen3-VL for visual extraction and ambiguous fields.

Optional engines:

- olmOCR for GPU-backed PDF/image-to-Markdown extraction.
- Surya if its model-weight license fits the deployment.

### 4. Layout Graph Builder

Responsibilities:

- Merge blocks from multiple OCR outputs.
- Cluster duplicate detections.
- Infer reading order.
- Detect labels, values, tables, checkboxes, signatures, stamps, and headers.
- Link labels to nearby values.

The layout graph is the central internal representation. It avoids relying on plain text only, because forms are spatial documents.

### 5. Extraction Engine

Responsibilities:

- Extract key-value pairs without fixed templates.
- Use geometry-based pairing.
- Use VLM-based extraction for ambiguous layouts.
- Normalize common field names.
- Preserve raw observed text.
- Avoid inventing values.

The extraction engine should return candidates first, then let validation decide which candidate becomes final.

### 6. Validator and Confidence Engine

Responsibilities:

- Score each candidate field.
- Compare model agreement.
- Validate field types.
- Detect conflicts and missing values.
- Trigger automatic retries.
- Assign final status.

Statuses:

- `accepted`: high confidence and no important conflict.
- `uncertain`: value exists but confidence is weak.
- `missing`: field detected but no value found.
- `conflict`: multiple plausible values disagree.
- `rejected`: likely hallucination, noise, or invalid value.

### 7. Exporter

Responsibilities:

- Write per-document outputs.
- Save evidence crops.
- Produce machine-readable and human-readable files.
- Keep raw model outputs for audit.

Outputs:

- `raw_text.md`
- `layout.json`
- `extracted_fields.json`
- `extracted_fields.csv`
- `confidence_report.json`
- `evidence/*.png`

## Data Flow Contract

Every stage must keep:

- `document_id`
- `page_id`
- `source_file`
- `bbox`
- `text`
- `confidence`
- `model`
- `preprocessing_variant`
- `evidence_path`

This makes the final result auditable without manual validation.

## Zero-Dollar Infrastructure

The architecture should run on:

- Local Windows machine with Python.
- Local Linux machine.
- Free GPU notebook when available.
- Existing in-house GPU machine.

It should not require:

- Paid OCR APIs.
- Paid hosted LLMs.
- Cloud storage.
- Manual annotation services.

## Privacy Model

Default privacy stance:

- Documents stay local.
- Model inference runs locally.
- Outputs are written to local disk.
- Logs must not include full sensitive documents unless explicitly enabled.

## Failure Handling

The system should not fail silently. For each document:

- If a page cannot render, mark document as `render_failed`.
- If OCR fails, keep image evidence and error logs.
- If extraction is ambiguous, return candidates with `uncertain` or `conflict`.
- If a value is missing, return `missing` rather than guessing.

