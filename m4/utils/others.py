# coding: utf-8
import torch
import argparse
import torch.nn as nn
import random
import json
import numpy as np
from typing import Dict, Union


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


def load_json(path: str):
    with open(path, 'r') as fin:
        json_dict = json.load(fin)
        args = argparse.Namespace(**json_dict)
    return args


def dump_json(data: Union[Dict, argparse.Namespace],
              path: str,
              indent: int=2
    ):
    if isinstance(data, argparse.Namespace):
        data = vars(data)
    with open(path, 'w') as fout:
        json_dict = json.dump(data, fout, indent=indent)


