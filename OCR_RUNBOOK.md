# OCR/VLM Operational Runbook

## Daily Batch Flow

1. Put source files in the input folder.
2. Run the extraction command.
3. Check batch summary.
4. Send accepted and uncertain outputs downstream.
5. Archive source files and outputs together.

Recommended command:

```powershell
python -m ocr_extract run --input data\input --output data\output --config configs\default.yaml
```

For the current workspace, the command can point directly at the current folder:

```powershell
python -m ocr_extract run --input . --output data\output --config configs\default.yaml
```

## Single Document Run

```powershell
python -m ocr_extract run --input AOF_01.pdf --output data\output\AOF_01
```

Expected output:

```text
data/output/AOF_01/
  raw_text.md
  layout.json
  extracted_fields.json
  extracted_fields.csv
  confidence_report.json
  evidence/
```

## Benchmark Run

```powershell
python -m ocr_extract benchmark --input . --output reports\benchmarks
```

Expected output:

```text
reports/benchmarks/model_comparison.csv
reports/benchmarks/field_agreement.json
reports/benchmarks/runtime_summary.csv
reports/benchmarks/error_taxonomy.md
```

## Output Review Without Manual Validation

The process does not require manual validation, but downstream systems should use status fields.

Recommended ingestion rules:

- Ingest `accepted` fields automatically.
- Ingest `uncertain` fields only into review-tolerant workflows.
- Do not ingest `conflict` as final truth.
- Treat `missing` as blank, not as extraction failure.
- Store `evidence_crop` paths for audit.

## Log Files

Expected logs:

```text
logs/pipeline.log
logs/errors.log
logs/model_runtime.log
```

Important log events:

- `DOCUMENT_STARTED`
- `DOCUMENT_COMPLETED`
- `PAGE_RENDER_FAILED`
- `OCR_ENGINE_FAILED`
- `FIELD_RETRY_STARTED`
- `FIELD_ACCEPTED`
- `FIELD_UNCERTAIN`
- `FIELD_CONFLICT`
- `VLM_VALUE_REJECTED`

## Troubleshooting

### Tesseract Not Found

Check:

```powershell
tesseract --version
```

Fix:

- Install Tesseract.
- Add install folder to `PATH`.
- Restart terminal.

### PDF Pages Do Not Render

Check:

```powershell
pdftoppm -v
```

Fix:

- Install Poppler.
- Add Poppler `bin` folder to `PATH`.
- Try PyMuPDF fallback rendering.

### GPU Out of Memory

Symptoms:

- Qwen3-VL fails during inference.
- Process exits during generation.

Fix:

- Use Qwen3-VL-2B instead of larger models.
- Use quantized model if available.
- Run VLM only on crops, not full pages.
- Reduce image resolution for VLM pass.
- Disable olmOCR.

### Handwriting Quality Is Weak

Fix:

- Increase PDF render DPI.
- Expand crop margins.
- Use original and contrast variants.
- Run TrOCR on line crops.
- Ask Qwen3-VL to read only the crop.

### Too Many Hallucinated Fields

Fix:

- Require bbox evidence for VLM fields.
- Lower VLM authority in confidence formula.
- Reject VLM-only fields unless the prompt returns localized evidence.
- Add stricter JSON schema.

### Wrong Label-Value Pairing

Fix:

- Increase geometry weight.
- Detect table rows before pairing.
- Ignore headers and footers.
- Penalize values that cross field boundaries.

## Release Checklist

Before using on real batches:

- All dependencies installed.
- One sample PDF renders correctly.
- PaddleOCR and Tesseract produce raw outputs.
- TrOCR runs on one handwritten crop.
- Qwen3-VL runs locally or is disabled in config.
- `extracted_fields.json` validates against schema.
- `extracted_fields.csv` opens correctly.
- Evidence crops are saved.
- Uncertain and conflict fields are not silently accepted.
- Logs contain no unintended sensitive full-page dumps.

## Production Guardrails

Keep these enabled:

- source file hashing
- raw model output retention
- evidence crops
- confidence report
- retry logs
- hallucination rejection
- validator warnings

Do not enable by default:

- cloud upload
- paid model APIs
- automatic deletion of source documents
- silent overwrite of previous outputs

## Recommended First Run on Workspace PDFs

Process all current PDFs:

```powershell
python -m ocr_extract run --input . --output data\output --config configs\default.yaml
```

Then inspect:

```powershell
Get-ChildItem data\output -Recurse -Filter confidence_report.json
Get-ChildItem data\output -Recurse -Filter extracted_fields.csv
```

The first run should be treated as a benchmark pass. Use its confidence reports to tune thresholds before automating downstream ingestion.

