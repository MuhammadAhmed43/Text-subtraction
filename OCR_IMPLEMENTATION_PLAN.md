# OCR/VLM Implementation Plan

## Objective

Implement a reusable extraction system for handwritten and mixed-layout forms using only open-source/open-weight tools and zero-dollar infrastructure.

The system must support documents with unknown layouts. It should not depend on hand-built templates, although optional domain schemas can be added to normalize output fields.

## Success Criteria

The implementation is successful when it can:

- Process all PDFs in a folder without manual page handling.
- Extract printed labels, handwritten values, tables, checkboxes, and signatures where visible.
- Produce JSON, CSV, Markdown, and field evidence crops.
- Assign confidence and uncertainty reasons for every field.
- Re-run low-confidence areas automatically using better crops or a stronger model.
- Batch process new documents with the same command.
- Keep all data local unless explicitly configured otherwise.

## Phase 1: Environment and Input Handling

Deliverables:

- Python virtual environment.
- System dependencies installed: Poppler, Tesseract, and optional Ghostscript.
- Project folder structure created.
- PDF-to-image conversion working at 300-400 DPI.
- Input manifest generated for all PDFs.

Implementation tasks:

- Scan input folder for `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff`.
- Assign stable `document_id` values using filename plus content hash.
- Render each PDF page to high-resolution images.
- Save original page image plus preprocessing variants.
- Track page size, DPI, rotation, and render settings in `manifest.json`.

Exit criteria:

- Every source document has page images in `work/pages/`.
- `work/manifests/document_manifest.json` lists every input and every generated page.

## Phase 2: Preprocessing

Deliverables:

- Deskewed page images.
- Contrast-enhanced variants.
- Binarized variants.
- Detected page orientation.
- Optional border and margin crop.

Implementation tasks:

- Detect orientation with OCR orientation tools and image heuristics.
- Apply deskew using Hough lines or projection profile analysis.
- Create variants:
  - `original`
  - `gray`
  - `contrast`
  - `binarized`
  - `denoised`
- Preserve mappings from each variant to the original page coordinates.

Exit criteria:

- Each page has multiple image variants available.
- Coordinate transforms can map every crop back to the original page.

## Phase 3: OCR Ensemble

Deliverables:

- PaddleOCR output.
- Tesseract output.
- Docling output.
- TrOCR output for detected handwritten crops.
- Optional Qwen3-VL or olmOCR output.

Implementation tasks:

- Run PaddleOCR on page variants and keep the best line-level result.
- Run Tesseract with TSV/hOCR output for baseline comparison.
- Run Docling for PDF and document structure where applicable.
- Detect likely handwritten regions using:
  - field boxes and underlines
  - label-value distance
  - connected components
  - low agreement between printed OCR engines
- Crop likely handwritten lines and run TrOCR.
- Run Qwen3-VL on full pages and difficult crops when GPU/RAM allows.

Exit criteria:

- `layout.json` contains all OCR blocks with model name, text, bbox, confidence, and page id.
- At least two independent model outputs exist for most readable text regions.

## Phase 4: Layout Graph

Deliverables:

- Page-level layout graph.
- Reading order.
- Detected label-value candidates.
- Table and checkbox regions.

Implementation tasks:

- Merge OCR blocks from all models into spatial clusters.
- Determine reading order using top-to-bottom and left-to-right geometry, with table-aware exceptions.
- Classify blocks as likely label, value, table cell, paragraph, header, footer, signature, stamp, checkbox, or noise.
- Link likely labels to nearby values using geometry:
  - right of label
  - below label
  - inside same row
  - nearest blank line or field box
- Keep multiple candidates rather than forcing a single answer too early.

Exit criteria:

- Each candidate field has label text, value text, bounding boxes, source models, and evidence crops.

## Phase 5: Schema-Free Extraction

Deliverables:

- JSON extraction without fixed templates.
- Optional normalized schema mapping.
- VLM-assisted extraction for ambiguous pages.

Implementation tasks:

- Use layout graph to generate field candidates.
- Ask the local VLM to extract structured JSON from:
  - page image
  - OCR text
  - candidate labels and values
  - coordinates
- Instruct the VLM to return only observed values and not infer missing values.
- Merge VLM fields with geometry-based candidates.
- Normalize common field names into canonical keys when possible.

Exit criteria:

- `extracted_fields.json` contains all extracted fields with confidence and evidence.

## Phase 6: Automated Validation

Deliverables:

- Confidence scoring.
- Validator library.
- Automatic retry rules.
- Uncertainty flags.

Implementation tasks:

- Score model agreement.
- Score OCR confidence.
- Score field-type validity.
- Score label-value geometry.
- Detect conflicting outputs.
- Re-run weak fields with:
  - higher DPI crop
  - alternate preprocessing
  - TrOCR if handwriting
  - Qwen3-VL if visual reasoning needed
- Mark fields as `accepted`, `uncertain`, `missing`, or `conflict`.

Exit criteria:

- No extracted value is returned without status, score, source, and evidence.

## Phase 7: Export and Packaging

Deliverables:

- Command line runner.
- Batch processing.
- JSON, CSV, Markdown, and evidence outputs.
- Reproducible configuration.

Implementation tasks:

- Implement a CLI such as:
  - `python -m ocr_extract run --input data/input --output data/output`
  - `python -m ocr_extract benchmark --input data/input --output reports/benchmark`
- Write logs to `logs/`.
- Write per-document output folders.
- Include a config file for model choices and thresholds.

Exit criteria:

- New PDFs can be dropped into the input folder and processed with one command.

## Phase 8: Benchmark and Hardening

Deliverables:

- Model comparison report.
- Error taxonomy.
- Performance report.
- Regression test set.

Implementation tasks:

- Run the same documents through every enabled model.
- Compare field-level agreement.
- Track failures by type:
  - unreadable handwriting
  - cropped page
  - low contrast
  - wrong orientation
  - table confusion
  - hallucinated VLM field
- Save representative difficult crops for future tests.

Exit criteria:

- The best default model stack is selected with documented evidence.
- Future model changes can be compared against saved results.

## Practical Build Order

Recommended order:

1. PDF rendering and preprocessing.
2. PaddleOCR plus Tesseract baseline.
3. Layout graph and raw text export.
4. TrOCR on handwritten crops.
5. Qwen3-VL extraction pass.
6. Consensus and validators.
7. CSV/JSON outputs.
8. Benchmarking and retry logic.

## Non-Goals

The first version should not:

- Use paid OCR APIs.
- Depend on a fixed template for every form.
- Hide uncertainty from downstream systems.
- Fine-tune models before measuring off-the-shelf performance.
- Store private documents outside the local machine.

