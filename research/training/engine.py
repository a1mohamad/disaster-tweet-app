from typing import Any

import torch


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    metrics: dict[str, Any],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Train the model for one epoch and return loss plus metric values."""
    model.train()
    total_loss = 0.0
    for m in metrics.values():
        m.reset()

    for batch in loader:
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += (loss.item() * inputs.size(0))

        probs = torch.sigmoid(logits)
        for m in metrics.values():
            m.update(probs, labels)

    total_loss /= len(loader.dataset)
    results = {k: m.compute().item() for k, m in metrics.items()}

    return total_loss, results


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    metrics: dict[str, Any],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluate the model without gradient tracking."""
    model.eval()
    total_loss = 0.0
    for m in metrics.values():
        m.reset()

    for batch in loader:
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        total_loss += (loss.item() * inputs.size(0))
        for m in metrics.values():
            m.update(probs, labels)

    total_loss /= len(loader.dataset)
    results = {k: m.compute().item() for k, m in metrics.items()}
    return total_loss, results
