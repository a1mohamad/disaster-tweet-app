import numpy as np
import os
import random


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed available random generators for repeatable inference behavior."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Strict cuDNN settings are slower but useful when exact repeatability matters.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Using STRICT Deterministic mode (Slower).")
    else:
        # Benchmark mode lets cuDNN choose fast kernels for the current hardware.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
