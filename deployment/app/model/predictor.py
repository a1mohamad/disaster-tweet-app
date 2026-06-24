import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import numpy as np

from app.utils.errors import ArtifactError, InferenceError, ModelError


class OnnxBackend:
    """ONNX Runtime inference backend."""

    name = "onnx"

    def __init__(self, config: object) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise ModelError(
                "ONNX_RUNTIME_UNAVAILABLE",
                "onnxruntime is not installed.",
                {"error": str(exc)},
            ) from exc

        providers = ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(
                str(config.ONNX_MODEL_PATH), providers=providers
            )
        except Exception as exc:
            raise ModelError(
                "ONNX_LOAD_FAILED",
                "Failed to load ONNX model.",
                {"path": str(config.ONNX_MODEL_PATH), "error": str(exc)},
            ) from exc

        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

    def predict_logit(self, input_ids: np.ndarray, input_length: np.ndarray) -> float:
        """Run the ONNX model and return the raw logit."""
        feeds = {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "input_length": np.asarray(input_length, dtype=np.int64),
        }
        feeds = {name: feeds[name] for name in self.input_names}
        output = self.session.run([self.output_name], feeds)[0]
        return float(np.asarray(output).reshape(-1)[0])


class TorchBackend:
    """PyTorch inference backend used directly or as an ONNX fallback."""

    name = "torch"

    def __init__(self, config: object, vocab_size: int) -> None:
        try:
            import torch
            from app.model.disaster_model import DisasterTwittsClassifier
        except Exception as exc:
            raise ModelError(
                "TORCH_UNAVAILABLE",
                "PyTorch is not installed.",
                {"error": str(exc)},
            ) from exc

        try:
            self.torch = torch
            self.model = DisasterTwittsClassifier.from_pretrained(config, vocab_size)
            self.device = getattr(self.model, "device", torch.device("cpu"))
        except FileNotFoundError as exc:
            raise ModelError(
                "MODEL_NOT_FOUND",
                "Model file not found.",
                {"path": str(config.MODEL_PATH)},
            ) from exc
        except Exception as exc:
            raise ModelError(
                "MODEL_LOAD_FAILED",
                "Failed to load PyTorch model.",
                {"error": str(exc)},
            ) from exc

    def predict_logit(self, input_ids: np.ndarray, input_length: np.ndarray) -> float:
        """Run the PyTorch model and return the raw logit."""
        tensor_ids = self.torch.as_tensor(
            input_ids, dtype=self.torch.long, device=self.device
        )
        tensor_length = self.torch.as_tensor(input_length, dtype=self.torch.long)
        with self.torch.no_grad():
            logits = self.model(tensor_ids, lengths=tensor_length)
        return float(logits.detach().cpu().reshape(-1)[0].item())


class DisasterTwittsPredictor:
    """High-level predictor that wraps backend selection, thresholding, and labels."""

    def __init__(
        self,
        backend: OnnxBackend | TorchBackend,
        threshold: float,
        label_mapping: dict[int, str] | None = None,
        warnings: list[dict[str, Any]] | None = None,
    ) -> None:
        self.backend = backend
        self.threshold = float(threshold)
        self.label_mapping = label_mapping
        self.warnings = warnings or []

    @property
    def backend_name(self) -> str:
        """Return the selected backend name for API responses and logs."""
        return self.backend.name

    @classmethod
    def from_config(cls, config: object, vocab_size: int) -> "DisasterTwittsPredictor":
        """Build a predictor from application configuration and artifact metadata."""
        warnings = []
        backend = cls._load_backend(config, vocab_size, warnings)
        threshold = cls._load_threshold(config, warnings)
        return cls(backend=backend, threshold=threshold, warnings=warnings)

    @classmethod
    def _load_backend(
        cls, config: object, vocab_size: int, warnings: list[dict[str, Any]]
    ) -> OnnxBackend | TorchBackend:
        """Select and initialize the configured inference backend."""
        requested = config.INFERENCE_BACKEND
        if requested not in {"auto", "onnx", "torch"}:
            raise ModelError(
                "INVALID_BACKEND",
                "INFERENCE_BACKEND must be one of: auto, onnx, torch.",
                {"value": requested},
            )

        if requested in {"auto", "onnx"}:
            if config.ONNX_MODEL_PATH.exists():
                try:
                    return OnnxBackend(config)
                except ModelError:
                    if requested == "onnx" or not config.ALLOW_TORCH_FALLBACK:
                        raise
                    warnings.append(
                        {
                            "warning_code": "ONNX_BACKEND_FAILED",
                            "message": "ONNX backend failed. Trying PyTorch fallback.",
                            "details": {"path": str(config.ONNX_MODEL_PATH)},
                        }
                    )
            elif requested == "onnx":
                raise ModelError(
                    "ONNX_MODEL_NOT_FOUND",
                    "ONNX model file not found.",
                    {"path": str(config.ONNX_MODEL_PATH)},
                )
            else:
                # In auto mode the service prefers ONNX, then falls back to PyTorch.
                warnings.append(
                    {
                        "warning_code": "ONNX_MODEL_NOT_FOUND",
                        "message": "ONNX model file not found. Trying PyTorch fallback.",
                        "details": {"path": str(config.ONNX_MODEL_PATH)},
                    }
                )

        if requested == "auto" and not config.ALLOW_TORCH_FALLBACK:
            raise ModelError(
                "NO_INFERENCE_BACKEND",
                "ONNX model is unavailable and PyTorch fallback is disabled.",
                {"onnx_path": str(config.ONNX_MODEL_PATH)},
            )

        if not config.MODEL_PATH.exists():
            raise ModelError(
                "MODEL_NOT_FOUND",
                "PyTorch model file not found.",
                {"path": str(config.MODEL_PATH)},
            )
        return TorchBackend(config, vocab_size)

    @staticmethod
    def _load_threshold(config: object, warnings: list[dict[str, Any]]) -> float:
        """Load the decision threshold, falling back to configured default when safe."""
        if config.THRESHOLD_JSON_PATH.exists():
            try:
                with open(config.THRESHOLD_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return float(data["threshold"])
                return float(data)
            except Exception as exc:
                raise ArtifactError(
                    "THRESHOLD_JSON_INVALID",
                    "Threshold JSON file is invalid.",
                    {"path": str(config.THRESHOLD_JSON_PATH), "error": str(exc)},
                ) from exc

        try:
            import torch

            threshold = torch.load(config.THRESHOLD_PATH, map_location="cpu")
            if isinstance(threshold, torch.Tensor):
                threshold = threshold.item()
            return threshold
        except FileNotFoundError:
            warnings.append(
                {
                    "warning_code": "THRESHOLD_NOT_FOUND",
                    "message": "Threshold file not found. Using default threshold.",
                    "details": {
                        "path": str(config.THRESHOLD_PATH),
                        "default": float(config.THRESHOLD),
                    },
                }
            )
            return config.THRESHOLD
        except Exception:
            try:
                import onnxruntime  # noqa: F401
            except Exception:
                raise ArtifactError(
                    "THRESHOLD_LOAD_FAILED",
                    "Failed to load threshold. Install PyTorch or set THRESHOLD.",
                    {"path": str(config.THRESHOLD_PATH)},
                )
            warnings.append(
                {
                    "warning_code": "THRESHOLD_LOAD_SKIPPED",
                    "message": "PyTorch is unavailable to read threshold artifact. Using configured threshold.",
                    "details": {
                        "path": str(config.THRESHOLD_PATH),
                        "default": float(config.THRESHOLD),
                    },
                }
            )
            return config.THRESHOLD

    def load_label_mapping(self, path: str | Path) -> dict[int, str]:
        """Load integer label ids to display names."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        except FileNotFoundError as exc:
            raise ArtifactError(
                "LABEL_MAPPING_NOT_FOUND",
                "Label mapping file not found.",
                {"path": str(path)},
            ) from exc
        except JSONDecodeError as exc:
            raise ArtifactError(
                "LABEL_MAPPING_INVALID",
                "Label mapping file is not valid JSON.",
                {"path": str(path)},
            ) from exc
        self.label_mapping = {int(k): v for k, v in mapping.items()}
        return self.label_mapping

    def predict(
        self,
        input_ids: np.ndarray,
        input_length: np.ndarray,
        return_label_name: bool = True,
    ) -> tuple[float, int, str] | tuple[float, int]:
        """Return probability, numeric label, and optionally the label name."""
        try:
            logit = self.backend.predict_logit(input_ids, input_length)
            prob = float(1.0 / (1.0 + np.exp(-logit)))
            label = 1 if prob >= self.threshold else 0
            if return_label_name:
                if self.label_mapping:
                    label_name = self.label_mapping.get(label, str(label))
                else:
                    label_name = str(label)
                return prob, label, label_name
            return prob, label
        except Exception as exc:
            raise InferenceError(
                "INFERENCE_FAILED",
                "Prediction failed.",
                {"backend": self.backend_name, "error": str(exc)},
            ) from exc
