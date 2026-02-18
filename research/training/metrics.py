from torchmetrics.classification import (BinaryAccuracy, 
                                         BinaryPrecision, 
                                         BinaryRecall, 
                                         BinaryF1Score)

def build_train_metrics(cfg):
    train_metrics = {
        "accuracy": BinaryAccuracy(threshold=cfg.METRIC_THRESHOLD).to(cfg.DEVICE)
    }

    return train_metrics

def build_val_metrics(cfg):

    val_metrics = {
        "accuracy": BinaryAccuracy(threshold=cfg.METRIC_THRESHOLD).to(cfg.DEVICE),
        "precision": BinaryPrecision(threshold=cfg.METRIC_THRESHOLD).to(cfg.DEVICE),
        "recall": BinaryRecall(threshold=cfg.METRIC_THRESHOLD).to(cfg.DEVICE),
        "f1": BinaryF1Score(threshold=cfg.METRIC_THRESHOLD).to(cfg.DEVICE)
    }

    return val_metrics