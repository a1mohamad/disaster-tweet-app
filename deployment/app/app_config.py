import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class AppConfig:
    # Reproducibility
    SEED = int(os.getenv("SEED", "28"))

    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(BASE_DIR / "artifacts")))
    DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    ONNX_MODEL_PATH = Path(
        os.getenv("ONNX_MODEL_PATH", str(ARTIFACTS_DIR / "best_model.onnx"))
    )
    MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ARTIFACTS_DIR / "best_model.pt")))
    VOCAB_PATH = Path(os.getenv("VOCAB_PATH", str(ARTIFACTS_DIR / "vocabs.json")))
    THRESHOLD_PATH = Path(
        os.getenv("THRESHOLD_PATH", str(ARTIFACTS_DIR / "best_threshold.pt"))
    )
    THRESHOLD_JSON_PATH = Path(
        os.getenv("THRESHOLD_JSON_PATH", str(ARTIFACTS_DIR / "best_threshold.json"))
    )
    LABEL_MAPPING_PATH = Path(
        os.getenv("LABEL_MAPPING_PATH", str(ARTIFACTS_DIR / "label_mapping.json"))
    )
    DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "predictions.db")))

    # Model Architecture (MUST match training)
    EMB_DIM = int(os.getenv("EMB_DIM", "100"))
    HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "64"))
    OUTPUT_DIM = int(os.getenv("OUTPUT_DIM", "1"))
    NUM_LAYERS = int(os.getenv("NUM_LAYERS", "2"))
    DROPOUT = float(os.getenv("DROPOUT", "0.28"))
    BIDIRECTIONAL = _env_bool("BIDIRECTIONAL", True)
    FREEZE_EMBEDDING = _env_bool("FREEZE_EMBEDDING", True)

    # Preprocessing
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "200"))
    USE_KEYWORD = _env_bool("USE_KEYWORD", True)
    LOWERCASE = _env_bool("LOWERCASE", True)
    UPPERCASE = _env_bool("UPPERCASE", False)
    STRIP_MULTIPLE_WHITESPACE = _env_bool("STRIP_MULTIPLE_WHITESPACE", True)

    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    # Inference
    THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
    INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "auto").strip().lower()
    ALLOW_TORCH_FALLBACK = _env_bool("ALLOW_TORCH_FALLBACK", True)
    DEVICE = os.getenv("DEVICE", "auto").strip().lower()
