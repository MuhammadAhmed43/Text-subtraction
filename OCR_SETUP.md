# OCR/VLM Local Setup

## Hardware Profiles

### CPU-Only Minimum

Use this when no GPU is available.

Recommended engines:

- PaddleOCR CPU mode.
- Tesseract.
- Docling standard pipeline.
- TrOCR small or base only for selected crops.

Expected behavior:

- Cheapest and simplest.
- Slower on batches.
- Good for printed labels and moderate handwriting crops.
- Weak for full-page VLM reasoning.

### Consumer GPU Recommended

Use this when an NVIDIA GPU is available.

Recommended engines:

- PaddleOCR GPU mode.
- TrOCR handwritten.
- Qwen3-VL-2B-Instruct or quantized Qwen3-VL.
- Optional olmOCR if VRAM is enough.

Expected behavior:

- Better handwriting and schema-free extraction.
- Faster batch processing.
- Stronger handling of ambiguous layouts.

### Free Notebook Option

Use free Kaggle or Colab only if local GPU is unavailable and document privacy permits upload.

Rules:

- Do not upload private customer forms unless approved.
- Prefer local execution for sensitive data.
- Cache models only within the notebook session.

## System Dependencies

Install these outside Python:

- Poppler: PDF rendering utilities.
- Tesseract OCR: baseline OCR engine.
- Ghostscript: optional PDF/image handling helper.
- Visual C++ Build Tools on Windows if a Python package requires native compilation.

Windows notes:

- Add Tesseract and Poppler binaries to `PATH`.
- Confirm these commands work:

```powershell
tesseract --version
pdftoppm -v
python --version
```

## Python Environment

Recommended Python:

- Python 3.10 or 3.11.

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## Core Python Packages

Install the implementation dependencies in groups.

Document and image utilities:

```powershell
pip install pymupdf pdf2image pillow opencv-python numpy pandas pydantic rich tqdm
```

OCR engines:

```powershell
pip install paddleocr pytesseract python-doctr docling
```

PyTorch and transformer models:

```powershell
pip install torch torchvision transformers accelerate sentencepiece protobuf safetensors
pip install qwen-vl-utils
```

Optional GPU packages depend on CUDA version. Install PyTorch from the official PyTorch selector for the target CUDA runtime.

## Model Sources

Recommended model sources:

- PaddleOCR: `https://github.com/PaddlePaddle/PaddleOCR`
- Tesseract: `https://github.com/tesseract-ocr/tesseract`
- TrOCR handwritten: `https://huggingface.co/microsoft/trocr-base-handwritten`
- Qwen3-VL-2B-Instruct: `https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct`
- Docling: `https://github.com/docling-project/docling`
- olmOCR: `https://github.com/allenai/olmocr`
- Surya: `https://github.com/datalab-to/surya`

License notes:

- Verify model-weight licenses before commercial deployment.
- Keep Surya optional because its model weights have usage restrictions.
- Prefer Apache-2.0 or MIT model/code licenses for the default stack.

## Recommended Project Structure

```text
ocr_project/
  data/
    input/
    output/
  work/
    pages/
    variants/
    crops/
    manifests/
    model_raw/
  reports/
    benchmarks/
    errors/
  logs/
  configs/
    default.yaml
  src/
    ocr_extract/
```

For the current workspace, the existing PDFs can be treated as `data/input/` until the codebase is created.

## Configuration Defaults

Recommended defaults:

```yaml
render:
  dpi: 350
  image_format: png
preprocess:
  enable_deskew: true
  enable_denoise: true
  variants:
    - original
    - gray
    - contrast
    - binarized
ocr:
  engines:
    - paddleocr
    - tesseract
    - doctr
    - trocr_handwritten
    - qwen_vl
validation:
  accept_threshold: 0.82
  uncertain_threshold: 0.55
  rerun_threshold: 0.70
outputs:
  save_evidence_crops: true
  save_raw_model_outputs: true
```

## Smoke Test

After setup, run a smoke test on one PDF:

```powershell
python -m ocr_extract run --input AOF_01.pdf --output data\output\AOF_01
```

Expected files:

```text
data/output/AOF_01/raw_text.md
data/output/AOF_01/layout.json
data/output/AOF_01/extracted_fields.json
data/output/AOF_01/extracted_fields.csv
data/output/AOF_01/confidence_report.json
data/output/AOF_01/evidence/
```

