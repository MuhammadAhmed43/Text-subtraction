# OCR/VLM Model Evaluation

## Default Model Stack

Use this stack first:

| Model/tool | Main role | Why use it | License note |
| --- | --- | --- | --- |
| PaddleOCR | OCR, layout, document parsing | Strong general OCR and structured output | Apache-2.0 code; verify model licenses |
| Tesseract | Baseline OCR | Fast, local, useful for printed labels and confidence comparison | Apache-2.0 |
| Docling | PDF/document conversion | Layout, tables, Markdown/JSON conversion | MIT code; verify model licenses |
| TrOCR handwritten | Handwritten crop recognition | Strong on line-level handwriting | MIT model card on Hugging Face |
| Qwen3-VL-2B-Instruct | VLM extraction and reasoning | Open-weight VLM with document/OCR capability | Apache-2.0 model card |

## Optional Models

| Model/tool | When to use | Constraint |
| --- | --- | --- |
| olmOCR | Strong PDF/image-to-Markdown on GPU | Needs recent NVIDIA GPU and enough VRAM |
| Surya | OCR/layout/table recognition | Model weights have commercial usage restrictions |
| Donut | OCR-free document understanding experiments | Better for trained document tasks than arbitrary handwriting |
| LayoutLMv3 | Fine-tuned document AI | Needs labeled data for best key-value extraction |

## Why Use Multiple Models

No single open-source OCR/VLM is perfect for arbitrary handwritten forms.

The ensemble helps because:

- Tesseract is predictable for printed text.
- PaddleOCR is stronger for modern OCR and layout.
- TrOCR is better for cropped handwriting.
- Qwen3-VL can reason about forms visually.
- Docling can preserve document structure.

The system should use agreement between models as evidence, not as a popularity vote only.

## Benchmark Method

For each document:

1. Render all pages at fixed DPI.
2. Run every enabled model.
3. Save raw outputs.
4. Build layout graph.
5. Extract fields.
6. Compare field-level agreement.
7. Record runtime and memory usage.
8. Save low-confidence crops.

Benchmark outputs:

```text
reports/benchmarks/model_comparison.csv
reports/benchmarks/field_agreement.json
reports/benchmarks/runtime_summary.csv
reports/benchmarks/error_taxonomy.md
```

## Metrics

Use these metrics:

- Page render success rate.
- OCR block count.
- Mean OCR confidence.
- Field extraction count.
- Accepted field count.
- Uncertain field count.
- Conflict field count.
- Missing field count.
- Average runtime per page.
- Retry rate.
- VLM hallucination rejection count.

If ground truth becomes available later, add:

- Character error rate.
- Word error rate.
- Field exact match.
- Field normalized match.
- Table cell F1.

## Model Selection Rules

Use PaddleOCR result when:

- printed text confidence is high
- bbox is accurate
- output agrees with Tesseract or Docling

Use Tesseract result when:

- printed labels are clear
- PaddleOCR splits text poorly
- TSV confidence is high

Use TrOCR result when:

- crop is a single handwritten line
- printed OCR engines disagree
- handwriting is inside a field

Use Qwen3-VL result when:

- full-page reasoning is needed
- checkbox groups need interpretation
- label-value pairing is ambiguous
- table or form structure is unclear

Reject or downgrade VLM result when:

- no bbox or evidence supports it
- value appears hallucinated
- it contradicts high-confidence OCR
- it extracts a field from the prompt instead of the document

## Failure Modes

### Handwriting Is Unreadable

Symptoms:

- TrOCR and Qwen disagree.
- Crop quality is low.
- Multiple plausible names/numbers exist.

Response:

- mark as `uncertain`
- save evidence crop
- include all candidate values

### Form Is Cropped

Symptoms:

- labels cut off
- values missing at page edges
- no matching field boxes

Response:

- mark affected fields as `missing` or `uncertain`
- log page crop warning

### Table Confusion

Symptoms:

- values assigned to wrong headers
- reading order jumps columns

Response:

- use table recognition
- enforce row/column geometry
- lower confidence when header mapping is weak

### VLM Hallucination

Symptoms:

- VLM returns clean fields not visible in OCR/image
- values look generic
- no coordinates support the field

Response:

- reject VLM-only field
- log hallucination guardrail event

### Low Contrast Scan

Symptoms:

- OCR block count is low
- handwriting faint
- page background uneven

Response:

- try contrast and denoised variants
- rerender at higher DPI
- crop and rerun

## Recommended First Benchmark

Run all workspace PDFs through:

- PaddleOCR
- Tesseract
- Docling
- TrOCR on detected crops
- Qwen3-VL on full page or hard crops if hardware allows

Then compare:

- accepted fields per document
- uncertain fields per document
- conflicts per document
- runtime per page
- evidence crop quality

The first production default should be selected from this benchmark, not from model reputation alone.

