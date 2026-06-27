"""Pipeline configuration loader."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    dpi: int = 350
    image_format: str = "png"


@dataclass
class PreprocessConfig:
    enable_deskew: bool = True
    enable_denoise: bool = True
    variants: List[str] = field(
        default_factory=lambda: ["original", "gray", "contrast", "binarized"]
    )


@dataclass
class OCRConfig:
    engines: List[str] = field(
        default_factory=lambda: ["paddleocr", "tesseract", "trocr_handwritten", "qwen_vl"]
    )
    trocr_model: str = "microsoft/trocr-base-handwritten"
    vlm_model: str = "qwen2.5vl:7b"
    paddle_lang: str = "en"
    easyocr_lang: str = "en"
    tesseract_lang: str = "eng"


@dataclass
class ValidationConfig:
    accept_threshold: float = 0.82
    uncertain_threshold: float = 0.55
    rerun_threshold: float = 0.70
    max_retries: int = 2


@dataclass
class OutputConfig:
    save_evidence_crops: bool = True
    save_raw_model_outputs: bool = True


@dataclass
class PipelineConfig:
    render: RenderConfig = field(default_factory=RenderConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    vlm_prompt: str = ""

    # Paths - set at runtime
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    work_path: Optional[Path] = None
    project_root: Optional[Path] = None


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Load configuration from YAML file, falling back to defaults."""
    cfg = PipelineConfig()

    if config_path and os.path.exists(config_path):
        logger.info("Loading config from %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if "render" in raw:
            cfg.render = RenderConfig(**raw["render"])
        if "preprocess" in raw:
            cfg.preprocess = PreprocessConfig(**raw["preprocess"])
        if "ocr" in raw:
            cfg.ocr = OCRConfig(**raw["ocr"])
        if "validation" in raw:
            cfg.validation = ValidationConfig(**raw["validation"])
        if "outputs" in raw:
            cfg.outputs = OutputConfig(**raw["outputs"])
        if "vlm_prompt" in raw:
            cfg.vlm_prompt = raw["vlm_prompt"]
    else:
        logger.info("Using default configuration")

    return cfg
