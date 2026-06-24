import json
from typing import Any


class AppError(Exception):
    """Base application error that can be serialized for API responses."""

    def __init__(
        self,
        error_type: str,
        error_code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.error_code = error_code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Return an API-friendly error payload."""
        payload = {
            "error_type": self.error_type,
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload

    def to_json(self) -> str:
        """Serialize the error payload as JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=True)


class InputError(AppError):
    """Raised when user-provided text cannot be accepted."""

    def __init__(
        self, error_code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__("input_error", error_code, message, details)


class ArtifactError(AppError):
    """Raised when model, vocabulary, or metadata artifacts are missing or invalid."""

    def __init__(
        self, error_code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__("artifact_error", error_code, message, details)


class ModelError(AppError):
    """Raised when a model backend cannot be selected or loaded."""

    def __init__(
        self, error_code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__("model_error", error_code, message, details)


class InferenceError(AppError):
    """Raised when inference fails after inputs and artifacts are valid."""

    def __init__(
        self, error_code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__("inference_error", error_code, message, details)
