# OCR/VLM Output Schemas

## Design Rules

Every extracted value must be traceable to:

- source document
- page
- bounding box
- model output
- evidence crop
- confidence score
- validation status

The output should support both unknown document structures and optional canonical schemas.

## Document Manifest

File:

```text
work/manifests/document_manifest.json
```

Schema:

```json
{
  "batch_id": "2026-06-27T13-30-00",
  "created_at": "2026-06-27T13:30:00+05:00",
  "documents": [
    {
      "document_id": "aof_01_8f14e45f",
      "source_file": "AOF_01.pdf",
      "source_hash": "sha256...",
      "file_type": "pdf",
      "page_count": 2,
      "status": "ready",
      "pages": [
        {
          "page_id": "aof_01_8f14e45f_p0001",
          "page_number": 1,
          "image_path": "work/pages/aof_01_8f14e45f/page_0001.png",
          "width": 2480,
          "height": 3508,
          "dpi": 350
        }
      ]
    }
  ]
}
```

## OCR Block

Stored inside:

```text
layout.json
```

Schema:

```json
{
  "block_id": "blk_000123",
  "document_id": "aof_01_8f14e45f",
  "page_id": "aof_01_8f14e45f_p0001",
  "page_number": 1,
  "model": "paddleocr",
  "preprocessing_variant": "contrast",
  "text": "Applicant Name",
  "bbox": {
    "x1": 120,
    "y1": 340,
    "x2": 480,
    "y2": 382
  },
  "confidence": 0.94,
  "block_type": "label",
  "reading_order": 12,
  "raw_ref": "work/model_raw/aof_01/paddleocr/page_0001.json"
}
```

## Layout Graph

Schema:

```json
{
  "document_id": "aof_01_8f14e45f",
  "pages": [
    {
      "page_id": "aof_01_8f14e45f_p0001",
      "nodes": [
        {
          "node_id": "node_001",
          "node_type": "label",
          "text": "Applicant Name",
          "bbox": {
            "x1": 120,
            "y1": 340,
            "x2": 480,
            "y2": 382
          },
          "source_blocks": ["blk_000123", "blk_000531"]
        }
      ],
      "edges": [
        {
          "from": "node_001",
          "to": "node_002",
          "edge_type": "likely_value_for",
          "score": 0.86
        }
      ]
    }
  ]
}
```

## Field Candidate

Schema:

```json
{
  "candidate_id": "cand_000045",
  "field_name_raw": "Applicant Name",
  "field_name_normalized": "applicant_name",
  "value_raw": "Muhammad Ali",
  "value_normalized": "Muhammad Ali",
  "field_type": "person_name",
  "page_number": 1,
  "label_bbox": {
    "x1": 120,
    "y1": 340,
    "x2": 480,
    "y2": 382
  },
  "value_bbox": {
    "x1": 510,
    "y1": 332,
    "x2": 1010,
    "y2": 390
  },
  "evidence_crop": "evidence/page_0001_cand_000045.png",
  "sources": [
    {
      "model": "trocr_handwritten",
      "text": "Muhammad Ali",
      "confidence": 0.81
    },
    {
      "model": "qwen3_vl",
      "text": "Muhammad Ali",
      "confidence": null
    }
  ],
  "geometry_score": 0.91,
  "validator_score": 0.90,
  "agreement_score": 0.84,
  "final_score": 0.86,
  "status": "accepted",
  "uncertainty_reason": null
}
```

## Extracted Fields

File:

```text
extracted_fields.json
```

Schema:

```json
{
  "document_id": "aof_01_8f14e45f",
  "source_file": "AOF_01.pdf",
  "processed_at": "2026-06-27T13:45:00+05:00",
  "pipeline_version": "0.1.0",
  "overall_status": "accepted_with_uncertainties",
  "overall_confidence": 0.83,
  "fields": [
    {
      "field_name": "applicant_name",
      "display_name": "Applicant Name",
      "value": "Muhammad Ali",
      "raw_value": "Muhammad Ali",
      "field_type": "person_name",
      "status": "accepted",
      "confidence": 0.86,
      "page_number": 1,
      "evidence_crop": "evidence/page_0001_cand_000045.png",
      "source_models": ["trocr_handwritten", "qwen3_vl"],
      "uncertainty_reason": null
    }
  ],
  "unmapped_fields": [
    {
      "field_name_raw": "Branch",
      "value": "Gulberg",
      "confidence": 0.73,
      "status": "uncertain"
    }
  ]
}
```

## Confidence Report

File:

```text
confidence_report.json
```

Schema:

```json
{
  "document_id": "aof_01_8f14e45f",
  "summary": {
    "accepted": 18,
    "uncertain": 4,
    "missing": 2,
    "conflict": 1,
    "rejected": 3
  },
  "model_runtime_seconds": {
    "paddleocr": 8.4,
    "tesseract": 3.2,
    "trocr_handwritten": 14.7,
    "qwen3_vl": 27.1
  },
  "retry_count": 9,
  "warnings": [
    {
      "code": "LOW_HANDWRITING_CONFIDENCE",
      "field_name": "nominee_name",
      "page_number": 2,
      "message": "Models disagreed on handwritten value."
    }
  ]
}
```

## CSV Export

File:

```text
extracted_fields.csv
```

Columns:

```text
document_id
source_file
field_name
display_name
value
raw_value
field_type
status
confidence
page_number
evidence_crop
source_models
uncertainty_reason
```

## Raw Text Markdown

File:

```text
raw_text.md
```

Format:

```markdown
# AOF_01.pdf

## Page 1

Applicant Name: Muhammad Ali
CNIC: 12345-1234567-1

## Page 2

...
```

## Evidence Crop Naming

Use stable names:

```text
evidence/page_<page_number>_<field_name>_<candidate_id>.png
```

Example:

```text
evidence/page_0001_applicant_name_cand_000045.png
```

