from typing import Any

import pandas as pd
import torch


def pos_class_weight(df: pd.DataFrame, config: Any) -> torch.Tensor:
    """Calculate BCE positive-class weight from the training label balance."""
    neg_num = (df["target"] == 0).sum()
    pos_num = (df["target"] == 1).sum()

    pos_weight = torch.tensor([neg_num / pos_num]).to(config.DEVICE)

    return pos_weight
