"""Optional local MobileNetV3-Small selector with one-time loading and fallback.

This module never downloads weights or datasets. Training assets are produced by
the explicit scripts under ``experiments/perception/representation``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from contracts.models import RepresentationInfo, RepresentationKind, RepresentationScores
from services.perception.models import Modalities, OCRResult
from services.perception.representation.selector import (
    HeuristicRepresentationSelector,
    RepresentationSelector,
    preserve_pixels_on_failure,
)

logger = logging.getLogger(__name__)

REPRESENTATION_CLASSES: tuple[RepresentationKind, ...] = (
    "text_sufficient",
    "structure_sufficient",
    "pixels_required",
    "mixed",
)


class MobileNetV3SmallSelector(RepresentationSelector):
    """Load a local MobileNet checkpoint at most once; degrade to OCR heuristics."""

    _cache_lock = Lock()
    _model_cache: dict[str, tuple[Any | None, Any, Any, str, str, str | None, str | None]] = {}
    _model_failures: dict[str, str] = {}

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.65,
        onnx_path: str | Path | None = None,
    ) -> None:
        model_root = Path(__file__).resolve().parents[3] / "models/perception"
        default_path = model_root / "mobilenet_v3_small.pth"
        default_onnx_path = model_root / "mobilenet_v3_small.onnx"
        self.model_path = Path(model_path) if model_path is not None else default_path
        self.onnx_path = Path(onnx_path) if onnx_path is not None else default_onnx_path
        self.confidence_threshold = confidence_threshold
        self._load_attempted = False
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._model_id: str | None = None
        self._backend: str | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._load_error: str | None = None
        self._fallback = HeuristicRepresentationSelector()

    @property
    def model_available(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def _load_once(self) -> None:
        if self._model is not None:
            return
        if self._load_attempted:
            raise RuntimeError(self._load_error or "model load previously failed")
        self._load_attempted = True
        with self._cache_lock:
            candidates = [("onnxruntime", self.onnx_path), ("pytorch", self.model_path)]
            errors: list[str] = []
            for backend, path in candidates:
                key = f"{backend}:{path.resolve()}"
                cached = self._model_cache.get(key)
                if cached is not None:
                    (
                        self._torch,
                        self._model,
                        self._device,
                        self._model_id,
                        self._backend,
                        self._input_name,
                        self._output_name,
                    ) = cached
                    return
                failure = self._model_failures.get(key)
                if failure is not None:
                    errors.append(failure)
                    continue
                if not path.is_file():
                    errors.append(f"FileNotFoundError: model not found: {path}")
                    continue
                try:
                    if backend == "onnxruntime":
                        import onnxruntime as ort  # type: ignore[import-untyped]

                        model = ort.InferenceSession(
                            str(path), providers=["CPUExecutionProvider"]
                        )
                        input_name = model.get_inputs()[0].name
                        output_name = model.get_outputs()[0].name
                        torch_module = None
                        device: Any = "cpu"
                    else:
                        import torch
                        from torch import nn
                        from torchvision import models

                        model = models.mobilenet_v3_small(weights=None)
                        model.classifier[-1] = nn.Linear(
                            model.classifier[-1].in_features,
                            len(REPRESENTATION_CLASSES),
                        )
                        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                        if checkpoint.get("classes") != list(REPRESENTATION_CLASSES):
                            raise ValueError(
                                "model checkpoint classes do not match representation contract"
                            )
                        model.load_state_dict(checkpoint["state_dict"], strict=True)
                        device = torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )
                        model.to(device)
                        model.eval()
                        torch_module = torch
                        input_name = None
                        output_name = None

                    with path.open("rb") as model_file:
                        digest = hashlib.file_digest(model_file, "sha256").hexdigest()[:16]
                    model_id = f"mobilenet_v3_small_{backend}:{digest}"
                    self._torch = torch_module
                    self._model = model
                    self._device = device
                    self._model_id = model_id
                    self._backend = backend
                    self._input_name = input_name
                    self._output_name = output_name
                    self._model_cache[key] = (
                        torch_module,
                        model,
                        device,
                        model_id,
                        backend,
                        input_name,
                        output_name,
                    )
                    return
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    errors.append(error)
                    self._model_failures[key] = error
            self._load_error = "; ".join(errors)
            raise RuntimeError(self._load_error or "no local model artifacts are available")

    def _predict(self, image_bytes: bytes) -> list[float]:
        self._load_once()
        if self._model is None:
            raise RuntimeError("model failed to initialize")

        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)

        if self._backend == "onnxruntime":
            import numpy as np

            pixels = np.asarray(rgb, dtype=np.float32) / 255.0
            mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
            std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
            tensor = ((pixels - mean) / std).transpose((2, 0, 1))[None, ...].copy()
            logits = self._model.run(
                [self._output_name], {self._input_name: tensor}
            )[0][0]
            shifted = [float(item) - max(float(value) for value in logits) for item in logits]
            exponentials = [math.exp(value) for value in shifted]
            normalizer = sum(exponentials)
            return [value / normalizer for value in exponentials]

        if self._torch is None:
            raise RuntimeError("PyTorch model backend failed to initialize")

        from torchvision.transforms.functional import (
            normalize,
            pil_to_tensor,
        )

        tensor = pil_to_tensor(rgb).float().div_(255.0)
        tensor = normalize(tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        with self._torch.inference_mode():
            logits = self._model(tensor.unsqueeze(0).to(self._device))
            probabilities = self._torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
        return [float(value) for value in probabilities]

    def score_image(self, image_bytes: bytes) -> list[float]:
        """Return class probabilities; exposed for reproducible offline benchmarks."""
        probabilities = self._predict(image_bytes)
        if len(probabilities) != len(REPRESENTATION_CLASSES):
            raise ValueError("model returned an unexpected number of class probabilities")
        if any(not math.isfinite(value) or value < 0 for value in probabilities):
            raise ValueError("model returned a non-finite or negative class probability")
        total = sum(probabilities)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("model returned invalid class probabilities")
        return [value / total for value in probabilities]

    def select(
        self,
        modalities: Modalities,
        primary_modality: str,
        overall_confidence: float,
        ocr: OCRResult,
        visual_required: bool,
        image_bytes: bytes,
    ) -> RepresentationInfo:
        heuristic = self._fallback.select(
            modalities,
            primary_modality,
            overall_confidence,
            ocr,
            visual_required,
            image_bytes,
        )
        try:
            probabilities = self.score_image(image_bytes)
        except Exception as exc:
            logger.warning("representation_model_unavailable_using_heuristic error=%s", exc)
            return preserve_pixels_on_failure(heuristic)

        confidence = max(probabilities)
        predicted = REPRESENTATION_CLASSES[probabilities.index(confidence)]
        # Keep pixels if the classifier is uncertain or conflicts with strong
        # visual/mixed evidence from the existing local perception signals.
        if confidence < self.confidence_threshold:
            predicted = "pixels_required"
        elif heuristic.kind in {"pixels_required", "mixed"} and predicted in {
            "text_sufficient",
            "structure_sufficient",
        }:
            predicted = "pixels_required" if heuristic.kind == "pixels_required" else "mixed"

        scores = RepresentationScores(
            **dict(zip(REPRESENTATION_CLASSES, probabilities, strict=True))
        )
        return RepresentationInfo(
            kind=predicted,
            scores=scores,
            confidence=confidence,
            pixel_dependency_prior=heuristic.pixel_dependency_prior,
            source=f"mobilenet_v3_small:{self._backend}",
            model=self._model_id,
            available_representations=heuristic.available_representations,
            local_capabilities={
                **heuristic.local_capabilities,
                "ml_model": True,
                "onnx_runtime": self._backend == "onnxruntime",
            },
        )
