from pathlib import Path

import numpy as np
import torch


def load_embedding(
    emb_dir: str | Path,
    idx2word: dict[str, int],
    emb_dim: int,
) -> torch.Tensor:
    """Load pretrained vectors into a matrix aligned with the project vocabulary."""
    vocab_size = len(idx2word)
    emb_matrix = np.random.normal(0.0, 1.0, (vocab_size, emb_dim))
    with open(emb_dir, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            if word in idx2word:
                idx = idx2word[word]
                emb_matrix[idx] = vector
    return torch.FloatTensor(emb_matrix)
