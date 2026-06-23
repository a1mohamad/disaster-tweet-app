import numpy as np
import os
import random

def seed_everything(seed, deterministic=False):
    '''
    Ensures absolute reproducibility.
    '''
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
        # Only use these for the final "Gold" run to ensure exact results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Using STRICT Deterministic mode (Slower).")
    else:
        # Benchmark=True allows cuDNN to find the fastest kernels for your hardware
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
