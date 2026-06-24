import numpy as np
import os
import random

import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Strict cuDNN settings are slower but useful for final reproducibility checks.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Using STRICT Deterministic mode (Slower).")
    else:
        # Benchmark mode lets cuDNN choose fast kernels for the current hardware.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print("Using PROTOTYPING mode (Faster).")
    print(f"For Reproducibility, Everything seeded with {seed}!")
