# OCR/VLM Processing Pipeline

## Pipeline Overview

```text
discover inputs
  -> render pages
  -> preprocess images
  -> run OCR engines
  -> detect handwritten regions
  -> run handwritten recognizer
  -> build layout graph
  -> extract fields
  -> validate and retry
  -> export results
```

## Step 1: Discover Inputs

Supported input types:

- `.pdf`
- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`

For every input:

- Calculate file hash.
- Create `document_id`.
- Store source path, file size, and detected type.
- Skip duplicate files unless forced.

## Step 2: Render Pages

For PDFs:

- Render at 300-400 DPI.
- Prefer 350 DPI as the default.
- Save each page as PNG.
- Preserve page number and dimensions.

For images:

- Copy to page image format.
- Normalize EXIF orientation.
- Assign page number `1`.

Output example:

```text
work/pages/AOF_01/page_0001.png
work/pages/AOF_01/page_0002.png
```

## Step 3: Preprocess Variants

Create image variants for each page:

- `original`: unchanged rendered page.
- `gray`: grayscale.
- `contrast`: contrast-limited adaptive histogram equalization.
- `binarized`: adaptive thresholding.
- `denoised`: light denoise without destroying handwriting.

Important rule:

Do not overwrite the original page. All crops and bboxes must map back to original coordinates.

## Step 4: Run Printed/Layout OCR

Run these engines on the strongest variants:

- PaddleOCR.
- Tesseract TSV/hOCR.
- Docling document conversion.

Store raw outputs before merging:

```text
work/model_raw/AOF_01/paddleocr/page_0001.json
work/model_raw/AOF_01/tesseract/page_0001.tsv
work/model_raw/AOF_01/docling/document.json
```

Each OCR block must include:

- text
- bbox
- confidence
- model
- page id
- preprocessing variant

## Step 5: Detect Handwritten Regions

Candidate handwritten regions come from:

- Empty boxes next to printed labels.
- Text on underlines.
- Text inside form fields.
- Areas where PaddleOCR and Tesseract disagree.
- Low-confidence OCR near a readable label.
- Connected components with cursive-like strokes.

Crop rules:

- Expand crop by 8-20 pixels on each side.
- Keep original and contrast variants.
- Preserve crop-to-page coordinate transform.
- Avoid over-cropping ascenders, descenders, and signatures.

Output example:

```text
work/crops/AOF_01/page_0001/field_004_original.png
work/crops/AOF_01/page_0001/field_004_contrast.png
```

## Step 6: Run Handwriting OCR

Run TrOCR on line-level crops.

Rules:

- Use only likely handwritten crops, not full pages.
- Do not ask TrOCR to read tables or dense paragraphs.
- Keep multiple crop variants if outputs differ.
- Store generated text and confidence proxy.

Confidence proxy can use:

- generation probability when available
- crop quality
- agreement with VLM or geometry extraction
- character-level plausibility

## Step 7: Run VLM Extraction

Use Qwen3-VL-2B-Instruct for:

- ambiguous handwritten values
- unknown form layouts
- checkbox reasoning
- field grouping
- full-page extraction when hardware allows

Prompt requirements:

- Return strict JSON.
- Extract only visible values.
- Do not infer missing fields.
- Include page number and evidence region when possible.
- Mark unreadable fields as `uncertain` or `unreadable`.

Recommended prompt shape:

```text
You are extracting data from a scanned form.
Use only visible information from the image and OCR text.
Return strict JSON with fields:
field_name, value, page, evidence_text, uncertainty_reason.
Do not guess missing values.
```

## Step 8: Build Layout Graph

Merge all blocks into spatial clusters.

Cluster keys:

- bbox overlap
- text similarity
- model agreement
- same page
- same preprocessing variant family

Graph nodes:

- text block
- label
- value
- table
- cell
- checkbox
- signature
- stamp
- image/noise

Graph edges:

- near
- right_of
- below
- inside
- same_row
- same_column
- likely_value_for

## Step 9: Extract Field Candidates

Use these strategies together:

- Label-to-right value pairing.
- Label-to-below value pairing.
- Table cell pairing.
- Checkbox group extraction.
- VLM JSON extraction.
- Repeated-header/footer removal.

Each field may have multiple candidates.

Do not discard candidates until validation.

## Step 10: Validate and Retry

For each candidate:

- Compute confidence.
- Check validators.
- Compare model agreement.
- Check geometry plausibility.
- Check if the evidence crop actually contains the value.

Automatic retry triggers:

- confidence below rerun threshold
- disagreement between models
- invalid date/phone/number format
- crop too small or too large
- likely handwriting read by printed OCR only

Retry actions:

- render crop at higher DPI
- try alternate preprocessing
- expand crop
- run TrOCR
- run VLM on crop
- rerun PaddleOCR on crop

## Step 11: Export

Per document:

```text
data/output/<document_id>/
  raw_text.md
  layout.json
  extracted_fields.json
  extracted_fields.csv
  confidence_report.json
  evidence/
```

Batch summary:

```text
data/output/batch_summary.csv
data/output/batch_confidence.json
reports/errors/
```

## Step 12: Logging

Log:

- document start and end
- page render failures
- OCR engine failures
- model runtime
- retry decisions
- final confidence distribution

Do not log:

- full sensitive documents by default
- base64 page images
- private identifiers outside output files

