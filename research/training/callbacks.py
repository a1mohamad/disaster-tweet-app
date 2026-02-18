import json
from pathlib import Path

class EarlyStopping:
    def __init__(self, patience=5, mode='min', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.should_stop = False
        self.best_score = None
        self.counter = 0

    def step(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.mode == 'min':
            improvement = self.best_score - current_score > self.min_delta
        else:
            improvement = current_score - self.best_score > self.min_delta

        if improvement:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

class TrainingHistory:
    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "lr": []
        }

    def update(self, train_loss, val_loss, train_metrics, val_metrics, optimizer):
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_accuracy"].append(train_metrics["accuracy"])
        self.history["val_accuracy"].append(val_metrics["accuracy"])
        self.history["val_precision"].append(val_metrics["precision"])
        self.history["val_recall"].append(val_metrics["recall"])
        self.history["val_f1"].append(val_metrics["f1"])
        self.history["lr"].append(optimizer.param_groups[0]["lr"])

    def save(self, path: str):
        """Save training history to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        print(f"Training history saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load training history from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No file found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls()
        obj.history = data
        return obj
