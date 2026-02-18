import json
import torch

from json import JSONDecodeError
from app.utils.errors import ArtifactError, ModelError, InferenceError

from app.model.disaster_model import DisasterTwittsClassifier


class DisasterTwittsPredictor:
    def __init__(self, model, threshold, label_mapping=None, warnings=None):
        self.model = model
        self.threshold = float(threshold)
        self.label_mapping = label_mapping
        self.warnings = warnings or []

    @classmethod
    def from_config(cls, config, vocab_size):
        warnings = []
        try:
            model = DisasterTwittsClassifier.from_pretrained(config, vocab_size)
        except FileNotFoundError as exc:
            raise ModelError(
                "MODEL_NOT_FOUND",
                "Model file not found.",
                {"path": str(config.MODEL_PATH)},
            ) from exc
        except Exception as exc:
            raise ModelError(
                "MODEL_LOAD_FAILED",
                "Failed to load model.",
                {"error": str(exc)},
            ) from exc

        try:
            threshold = torch.load(config.THRESHOLD_PATH, map_location="cpu")
        except FileNotFoundError as exc:
            threshold = config.THRESHOLD
            warnings.append({
                "warning_code": "THRESHOLD_NOT_FOUND",
                "message": "Threshold file not found. Using default threshold.",
                "details": {"path": str(config.THRESHOLD_PATH), "default": float(config.THRESHOLD)},
            })
        except Exception as exc:
            raise ArtifactError(
                "THRESHOLD_LOAD_FAILED",
                "Failed to load threshold.",
                {"error": str(exc)},
            ) from exc
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.item()
        return cls(model=model, threshold=threshold, warnings=warnings)

    def load_label_mapping(self, path):
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

    @torch.no_grad()
    def predict(self, input_ids, input_length, return_label_name=True):
        try:
            logits = self.model(input_ids, lengths=input_length)
            prob = torch.sigmoid(logits).item()
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
                {"error": str(exc)},
            ) from exc
