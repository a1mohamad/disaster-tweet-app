'''
Updated AppConfig with model architecture hyperparameters.
'''
import os
from pathlib import Path
import torch

class AppConfig:
    # Reproducibility
    SEED = 28
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    DATA_DIR = BASE_DIR / "data"
    MODEL_PATH = ARTIFACTS_DIR / "best_model.pt"
    VOCAB_PATH = ARTIFACTS_DIR / "vocabs.json"
    THRESHOLD_PATH = ARTIFACTS_DIR / "best_threshold.pt"
    LABEL_MAPPING_PATH = ARTIFACTS_DIR / "label_mapping.json"
    DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "predictions.db")))

    # Model Architecture (MUST match training)
    EMB_DIM = 100       # e.g., if using glove.6B.100d
    HIDDEN_DIM = 64
    OUTPUT_DIM = 1      # Binary classification
    NUM_LAYERS = 2
    DROPOUT = 0.28
    BIDIRECTIONAL = True
    FREEZE_EMBEDDING = True

    # Preprocessing
    MAX_LENGTH = 200
    USE_KEYWORD = True
    LOWERCASE = True
    UPPERCASE = False
    STRIP_MULTIPLE_WHITESPACE = True

    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    # Inference
    THRESHOLD = 0.5  # Default threshold for classification
