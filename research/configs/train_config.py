from pathlib import Path
import torch


class TrainConfig:
    # -------------------- Reproducibility --------------------
    SEED = 28
    NUM_WORKERS = 0

    # -------------------- Tokens & Vocabulary --------------------
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1
    VOCAB_SIZE = 10000

    # -------------------- Text Processing --------------------
    LOWERCASE = True
    UPPERCASE = False
    STRIP_MULTIPLE_WHITESPACE = True
    USE_KEYWORD = True
    DROP_LOCATION = True
    MAX_LENGTH = 200

    # -------------------- Model Architecture --------------------
    BIDIRECTIONAL = True
    FREEZE_EMBEDDING = True
    EMB_DIM = 100
    HIDDEN_DIM = 64
    NUM_LSTM_LAYERS = 2
    DROPOUT = 0.28
    OUTPUT_DIM = 1   # binary classification

    # -------------------- Training --------------------
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 3e-4
    TRAIN_TEST_SPLIT = 0.8

    # -------------------- Early Stopping --------------------
    EARLY_STOP_PATIENCE = 5
    EARLY_STOP_MIN_DELTA = 1e-3

    # -------------------- Thresholds (training metrics) --------------------
    METRIC_THRESHOLD = 0.5

    # -------------------- Paths (training artifacts only) --------------------
    # Paths
    TRAIN_CSV = "./data/train.csv"
    GLOVE_PATH = "./embeddings/glove.6B.100d.txt"
    OUTPUT_DIR = Path("outputs")
    MODEL_PATH = OUTPUT_DIR / "best_model.pt"
    VOCAB_PATH = OUTPUT_DIR / "vocabs.json"
    HISTORY_PATH = OUTPUT_DIR / "training_history.json"

    # -------------------- Device --------------------
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    DEVICE = get_device.__func__()
