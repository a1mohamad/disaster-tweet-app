from torchmetrics.classification import (BinaryAccuracy, 
                                         BinaryPrecision, 
                                         BinaryRecall, 
                                         BinaryF1Score)


def build_train_metrics(cfg: object) -> dict[str, BinaryAccuracy]:
    """Create metrics tracked during training batches."""
    train_metrics = {
        "accuracy": BinaryAccuracy(threshold=cfg.METRIC_THRESHOLD).to(cfg.DEVICE)
    }

    return train_metrics


def build_val_metrics(cfg: object) -> dict[str, object]:
    """Create metrics tracked during validation batches."""

    val_metrics = {
        "accuracy": BinaryAccuracy(threshold=cfg.METRIC_THRESHOLD).to(cfg.DEVICE),
        "precision": BinaryPrecision(threshold=cfg.METRIC_THRESHOLD).to(cfg.DEVICE),
        "recall": BinaryRecall(threshold=cfg.METRIC_THRESHOLD).to(cfg.DEVICE),
        "f1": BinaryF1Score(threshold=cfg.METRIC_THRESHOLD).to(cfg.DEVICE)
    }

    return val_metrics
