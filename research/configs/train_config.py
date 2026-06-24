from pathlib import Path
import torch


class TrainConfig:
    """Central configuration for preprocessing, model architecture, and training."""

    # -------------------- Reproducibility --------------------
    SEED: int = 28
    NUM_WORKERS: int = 0

    # -------------------- Tokens & Vocabulary --------------------
    PAD_TOKEN: str = "<PAD>"
    UNK_TOKEN: str = "<UNK>"
    PAD_IDX: int = 0
    UNK_IDX: int = 1
    VOCAB_SIZE: int = 10000

    # -------------------- Text Processing --------------------
    LOWERCASE: bool = True
    UPPERCASE: bool = False
    STRIP_MULTIPLE_WHITESPACE: bool = True
    USE_KEYWORD: bool = True
    DROP_LOCATION: bool = True
    MAX_LENGTH: int = 200

    # -------------------- Model Architecture --------------------
    BIDIRECTIONAL: bool = True
    FREEZE_EMBEDDING: bool = True
    EMB_DIM: int = 100
    HIDDEN_DIM: int = 64
    NUM_LSTM_LAYERS: int = 2
    DROPOUT: float = 0.28
    OUTPUT_DIM: int = 1   # binary classification

    # -------------------- Training --------------------
    BATCH_SIZE: int = 128
    EPOCHS: int = 50
    LEARNING_RATE: float = 3e-4
    TRAIN_TEST_SPLIT: float = 0.8

    # -------------------- Early Stopping --------------------
    EARLY_STOP_PATIENCE: int = 5
    EARLY_STOP_MIN_DELTA: float = 1e-3

    # -------------------- Thresholds (training metrics) --------------------
    METRIC_THRESHOLD: float = 0.5

    # -------------------- Paths (training artifacts only) --------------------
    TRAIN_CSV: str = "./data/train.csv"
    GLOVE_PATH: str = "./embeddings/glove.6B.100d.txt"
    OUTPUT_DIR: Path = Path("outputs")
    MODEL_PATH: Path = OUTPUT_DIR / "best_model.pt"
    VOCAB_PATH: Path = OUTPUT_DIR / "vocabs.json"
    HISTORY_PATH: Path = OUTPUT_DIR / "training_history.json"

    # -------------------- Device --------------------
    @staticmethod
    def get_device() -> torch.device:
        """Return CUDA when available, otherwise CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    DEVICE: torch.device = get_device.__func__()
