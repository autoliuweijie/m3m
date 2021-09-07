# coding: utf-8
import os
import torch
import torch.nn as nn
import logging
from typing import List, Union, Optional, Dict
from ..model_base import HFImageModel
from argparse import Namespace
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image


class ViT(HFImageModel):

    def __init__(self,
                 model_dir: Union[str, os.PathLike]
        ):
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_dir)
        encoder = ViTModel.from_pretrained(model_dir)
        super(ViT,  self).__init__(model_dir, feature_extractor, encoder)

