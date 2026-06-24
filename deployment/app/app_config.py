import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with common truthy values."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class AppConfig:
    """Environment-driven configuration for the FastAPI inference service."""

    # Reproducibility
    SEED: int = int(os.getenv("SEED", "28"))

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    ARTIFACTS_DIR: Path = Path(os.getenv("ARTIFACTS_DIR", str(BASE_DIR / "artifacts")))
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    ONNX_MODEL_PATH: Path = Path(
        os.getenv("ONNX_MODEL_PATH", str(ARTIFACTS_DIR / "best_model.onnx"))
    )
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(ARTIFACTS_DIR / "best_model.pt")))
    VOCAB_PATH: Path = Path(os.getenv("VOCAB_PATH", str(ARTIFACTS_DIR / "vocabs.json")))
    THRESHOLD_PATH: Path = Path(
        os.getenv("THRESHOLD_PATH", str(ARTIFACTS_DIR / "best_threshold.pt"))
    )
    THRESHOLD_JSON_PATH: Path = Path(
        os.getenv("THRESHOLD_JSON_PATH", str(ARTIFACTS_DIR / "best_threshold.json"))
    )
    LABEL_MAPPING_PATH: Path = Path(
        os.getenv("LABEL_MAPPING_PATH", str(ARTIFACTS_DIR / "label_mapping.json"))
    )
    DB_PATH: Path = Path(os.getenv("DB_PATH", str(DATA_DIR / "predictions.db")))

    # Model Architecture (MUST match training)
    EMB_DIM: int = int(os.getenv("EMB_DIM", "100"))
    HIDDEN_DIM: int = int(os.getenv("HIDDEN_DIM", "64"))
    OUTPUT_DIM: int = int(os.getenv("OUTPUT_DIM", "1"))
    NUM_LAYERS: int = int(os.getenv("NUM_LAYERS", "2"))
    DROPOUT: float = float(os.getenv("DROPOUT", "0.28"))
    BIDIRECTIONAL: bool = _env_bool("BIDIRECTIONAL", True)
    FREEZE_EMBEDDING: bool = _env_bool("FREEZE_EMBEDDING", True)

    # Preprocessing
    MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "200"))
    USE_KEYWORD: bool = _env_bool("USE_KEYWORD", True)
    LOWERCASE: bool = _env_bool("LOWERCASE", True)
    UPPERCASE: bool = _env_bool("UPPERCASE", False)
    STRIP_MULTIPLE_WHITESPACE: bool = _env_bool("STRIP_MULTIPLE_WHITESPACE", True)

    # Special tokens
    PAD_TOKEN: str = "<PAD>"
    UNK_TOKEN: str = "<UNK>"
    PAD_IDX: int = 0
    UNK_IDX: int = 1

    # Inference
    THRESHOLD: float = float(os.getenv("THRESHOLD", "0.5"))
    INFERENCE_BACKEND: str = os.getenv("INFERENCE_BACKEND", "auto").strip().lower()
    ALLOW_TORCH_FALLBACK: bool = _env_bool("ALLOW_TORCH_FALLBACK", True)
    DEVICE: str = os.getenv("DEVICE", "auto").strip().lower()
