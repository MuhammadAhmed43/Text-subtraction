# Automated Validation and Confidence

## Core Principle

The system should avoid manual validation, but it must not pretend uncertain data is certain.

Every field gets:

- value
- status
- confidence score
- evidence crop
- model sources
- uncertainty reason when needed

## Field Statuses

### accepted

Use when:

- confidence is above the accept threshold
- validators pass
- no major model conflict exists
- evidence crop contains the value

### uncertain

Use when:

- a value exists but confidence is moderate
- handwriting is hard to read
- validators partially pass
- model agreement is weak

### missing

Use when:

- a label is found but no value is visible
- the form field appears blank
- the page is cropped before the value area

### conflict

Use when:

- multiple plausible values disagree
- OCR and VLM outputs are materially different
- two labels may point to the same value

### rejected

Use when:

- a candidate is likely noise
- the VLM invents a field not supported by OCR or evidence
- validator failure is severe
- value is outside the evidence crop

## Confidence Formula

Suggested weighted score:

```text
final_score =
  0.30 * model_agreement_score +
  0.20 * ocr_confidence_score +
  0.20 * geometry_score +
  0.20 * validator_score +
  0.10 * crop_quality_score
```

Default thresholds:

```text
accepted: 0.82 to 1.00
uncertain: 0.55 to 0.81
rejected or missing: below 0.55, depending on evidence
automatic retry: below 0.70 when a value seems present
```

## Model Agreement Score

Score higher when:

- multiple engines return the same text
- OCR and VLM agree
- TrOCR agrees with Qwen3-VL for handwriting
- repeated fields match across pages

Score lower when:

- models return different numbers
- one model returns a value and others see blank
- VLM output has no nearby OCR evidence
- text appears only in a poor-quality crop

## OCR Confidence Score

Use native confidence where available:

- PaddleOCR confidence.
- Tesseract word confidence.
- Docling confidence if available.
- TrOCR generation probability when available.

If a model does not expose confidence, estimate confidence using:

- agreement with other models
- crop sharpness
- text plausibility
- language model likelihood
- validator success

## Geometry Score

Score higher when:

- value is directly right of label
- value is directly below label
- value is inside the expected field box
- value is in the same table row
- checkbox mark is inside the box

Score lower when:

- value is far from label
- value crosses unrelated regions
- label-value line intersects another field
- crop includes multiple unrelated values

## Validator Score

Use field-specific validators only after field type is inferred.

Common validators:

- Date: valid calendar date, known formats, no impossible year.
- Phone: country-specific phone pattern.
- CNIC/National ID: digit count and separator pattern where applicable.
- Email: standard email syntax.
- Amount: numeric with optional commas and decimals.
- Account number: digit count and allowed separators.
- Name: alphabetic words with reasonable length.
- Address: multi-token free text, weak validation only.
- Checkbox: selected/unselected/ambiguous based on visual mark.

Important rule:

Validators should reduce confidence, not erase evidence. A handwritten value may be valid even if formatting is unusual.

## Crop Quality Score

Estimate:

- sharpness
- contrast
- crop size
- text density
- margin around text
- whether text is cut off

Retry if:

- crop is too tight
- crop is too blurry
- crop has too many unrelated fields
- crop does not include the full baseline

## Automatic Retry Rules

Run retry when:

- final score is below `0.70`
- a field looks present but unreadable
- model outputs conflict
- validator fails but evidence suggests the value may be real

Retry actions:

- expand crop
- use original image instead of thresholded variant
- use contrast variant instead of original
- render page at higher DPI
- run TrOCR on line crop
- run Qwen3-VL on crop
- rerun PaddleOCR on crop

Stop retry when:

- score crosses accept threshold
- all variants produce the same uncertainty
- max retry count is reached
- crop quality cannot be improved

## Hallucination Guardrails

VLMs can invent fields. Prevent that by requiring:

- visible evidence crop
- nearby OCR text or handwritten crop
- bbox or page reference
- no extraction from prompt examples
- strict JSON output

Reject VLM-only values when:

- value cannot be localized
- value is not visible in the image
- value contradicts OCR and validators
- VLM fills a blank field

## Blank Field Detection

A blank field should be returned as `missing`, not `uncertain`.

Signals for blank:

- printed label exists
- field box or underline exists
- no ink strokes inside value region
- OCR detects no value
- VLM reports blank

Signals for not blank:

- connected components inside field
- faint handwriting
- OCR detects low-confidence text
- visible strike or checkbox mark

## Checkbox Validation

Checkbox statuses:

- `checked`
- `unchecked`
- `ambiguous`

Use:

- mark density inside box
- diagonal/cross/tick line detection
- comparison with empty boxes
- VLM confirmation for ambiguous cases

## Signature and Stamp Handling

Signatures and stamps are usually not text fields.

Return:

- `present`
- `absent`
- `ambiguous`

Save evidence crop and do not force OCR text unless readable printed text exists.

