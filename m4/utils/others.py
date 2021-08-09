# coding: utf-8
import torch
import torch.nn as nn
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(x, device):
    if isinstance(x, dict):
        for key, item in x.items():
            if isinstance(item, torch.Tensor):
                x[key] = item.to(device)
    elif isinstance(x, torch.Tensor):
        x = x.to(device)
    return x
