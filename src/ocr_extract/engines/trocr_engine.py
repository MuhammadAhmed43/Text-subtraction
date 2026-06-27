"""TrOCR handwritten text recognition engine.

Provides functions to load Microsoft's TrOCR model and run inference
on cropped handwriting images, returning recognized text with confidence.
"""

import logging
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Module-level cache to avoid reloading the model on every call.
_cached_processor: Any = None
_cached_model: Any = None
_cached_model_name: str = ""


def load_trocr_model(
    model_name: str = "microsoft/trocr-base-handwritten",
) -> Tuple[Any, Any]:
    """Load TrOCR processor and model, caching them at module level.

    Args:
        model_name: HuggingFace model identifier for TrOCR.

    Returns:
        Tuple of (processor, model) ready for inference.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    global _cached_processor, _cached_model, _cached_model_name

    # Return cached instances if the same model was already loaded.
    if (
        _cached_processor is not None
        and _cached_model is not None
        and _cached_model_name == model_name
    ):
        logger.debug("Returning cached TrOCR model: %s", model_name)
        return _cached_processor, _cached_model

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        logger.info("Loading TrOCR processor from '%s' ...", model_name)
        processor = TrOCRProcessor.from_pretrained(model_name)

        logger.info("Loading TrOCR model from '%s' ...", model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # Move to GPU when available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info("TrOCR model loaded on %s", device)

        # Cache for subsequent calls.
        _cached_processor = processor
        _cached_model = model
        _cached_model_name = model_name

        return processor, model

    except Exception:
        logger.exception("Failed to load TrOCR model '%s'", model_name)
        raise RuntimeError(
            f"Could not load TrOCR model '{model_name}'"
        ) from None


def run_trocr_on_crop(
    image_path: str,
    processor: Any = None,
    model: Any = None,
    model_name: str = "microsoft/trocr-base-handwritten",
) -> Dict[str, Any]:
    """Run TrOCR inference on a single cropped image.

    Args:
        image_path: Path to the cropped image file.
        processor: Pre-loaded TrOCRProcessor (optional).
        model: Pre-loaded VisionEncoderDecoderModel (optional).
        model_name: HuggingFace model identifier used when
            ``processor`` / ``model`` are not supplied.

    Returns:
        Dict with keys ``text``, ``confidence``, and ``model``.
    """
    try:
        # Load model if not provided.
        if processor is None or model is None:
            processor, model = load_trocr_model(model_name)

        # Open and convert the image to RGB.
        image = Image.open(image_path).convert("RGB")
        logger.debug("Opened image: %s", image_path)

        # Determine the device the model lives on.
        device = next(model.parameters()).device

        # Prepare pixel values.
        pixel_values = processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(device)

        # Generate text with scores so we can derive confidence.
        outputs = model.generate(
            pixel_values,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Decode the generated token ids.
        generated_ids = outputs.sequences
        text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # Derive a confidence value from generation scores when available.
        confidence = _extract_confidence(outputs)

        logger.info(
            "TrOCR result for '%s': text='%s', confidence=%.4f",
            image_path,
            text,
            confidence,
        )

        return {
            "text": text,
            "confidence": confidence,
            "model": "trocr_handwritten",
        }

    except Exception:
        logger.exception(
            "Error running TrOCR on '%s'", image_path
        )
        return {
            "text": "",
            "confidence": 0.0,
            "model": "trocr_handwritten",
        }


def run_trocr_on_crops(
    crop_paths: List[str],
    model_name: str = "microsoft/trocr-base-handwritten",
) -> List[Dict[str, Any]]:
    """Run TrOCR inference on a batch of cropped images.

    The model is loaded once and reused for every crop in the list.

    Args:
        crop_paths: List of file paths to cropped images.
        model_name: HuggingFace model identifier for TrOCR.

    Returns:
        List of result dicts (same schema as :func:`run_trocr_on_crop`).
    """
    results: List[Dict[str, Any]] = []

    try:
        processor, model = load_trocr_model(model_name)
    except RuntimeError:
        logger.error(
            "Cannot process crops - model failed to load."
        )
        return [
            {"text": "", "confidence": 0.0, "model": "trocr_handwritten"}
            for _ in crop_paths
        ]

    for path in crop_paths:
        result = run_trocr_on_crop(
            image_path=path,
            processor=processor,
            model=model,
            model_name=model_name,
        )
        results.append(result)

    logger.info(
        "Processed %d crops with TrOCR (%d succeeded).",
        len(crop_paths),
        sum(1 for r in results if r["text"]),
    )
    return results


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _extract_confidence(outputs: Any) -> float:
    """Compute a rough confidence score from generation outputs.

    Uses the mean of per-step log-probabilities converted to a
    probability.  Falls back to 0.5 when scores are unavailable.

    Args:
        outputs: The ``GenerateOutput`` returned by ``model.generate``.

    Returns:
        A float in [0, 1] representing confidence.
    """
    try:
        if hasattr(outputs, "scores") and outputs.scores:
            import torch.nn.functional as F

            log_probs: List[float] = []
            for score in outputs.scores:
                probs = F.softmax(score, dim=-1)
                max_prob = probs.max(dim=-1).values.item()
                log_probs.append(max_prob)

            if log_probs:
                confidence = sum(log_probs) / len(log_probs)
                return float(confidence)
    except Exception:
        logger.debug(
            "Could not extract confidence from generation scores."
        )

    return 0.5
