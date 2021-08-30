# coding: utf-8
import os
import shutil
import time
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Union, List, Dict
from PIL import Image


class BaseModel(nn.Module):
    """
    Abstract Class for all models.

    @ref: https://zhuanlan.zhihu.com/p/319810661
    """

    def __init__(self,
                 model_dir: Union[str, os.PathLike]
        ):
        super().__init__()
        self.model_dir = model_dir
        self.device    = torch.device('cpu')

    def save_model(self,
                   model_dir: Union[str, os.PathLike],
                   epoch_num: int
        ):
        raise NotImplementedError

    def forward(self,
                *args,
                **kwargs,
        ):
        raise NotImplementedError

    def to(self,
           device):
        self.device = device
        super().to(device)

    def training_step(self,
                      batch: Dict[str, List],
                      batch_idx: int
        ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def training_step_end(self,
                          batch_parts: List
        ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def training_epoch_end(self,
                           training_step_outputs: List
        ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def training_step(self,
                      batch: Dict[str, List],
                      batch_idx: int
        ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def training_step_end(self,
                          batch_parts: List
        ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def training_epoch_end(self,
                           training_step_outputs: List
        ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def load_from_online(self,
                         model_name: str
        ):
        raise NotImplementedError


class UERModel(BaseModel):
    """
    Abstract class for all models whose tokenizer and encoder are implemented
    with UER-py.
    """
    pass


class HFModel(BaseModel):
    """
    Abstract class for all models whose tokenizer and encoder are implemented
    with huggingface transformers.
    """
    pass


class HFTextModel(HFModel):
    """
    Abstract class for all text models whose tokenizer and encoder are implemented
    with huggingface transformers.
    """

    def __init__(self,
                 model_dir : Union[str, os.PathLike],
                 tokenizer : nn.Module,
                 encoder   : nn.Module,
                 max_length: int
        ):
        super().__init__(model_dir)
        self.tokenizer  = tokenizer
        self.encoder    = encoder
        self.max_length = max_length

    def save_model(self,
                   model_dir: Union[str, os.PathLike],
                   epoch_num: int = None,
                   step_num : int = None
        ):
        if model_dir.endswith('/'):
            model_dir = model_dir.strip('/')
        if epoch_num is not None:
            model_dir = model_dir + '-epoch-{}'.format(epoch_num)
        if step_num is not None:
            model_dir = model_dir + '-step-{}'.format(step_num)
        if os.path.exists(model_dir):
            pass
        else:
            os.mkdir(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        self.encoder.save_pretrained(model_dir)
        return model_dir

    def forward(self,
                input_text_batch : List[str],
                input_text2_batch: List[str] = None,
                padding          : bool = True,
                truncation       : bool = True,
        ) -> Dict[str, Dict]:
        tokenize_res = self.tokenize(input_text_batch, input_text2_batch, padding, truncation)
        encode_res   = self.encode(tokenize_res)
        output_dict  = {
            'tokenize_res': tokenize_res,
            'encode_res'  : encode_res
        }
        return output_dict

    def tokenize(self,
                 input_text_batch: List[str],
                 input_text2_batch: List[str] = None,
                 padding         : bool = True,
                 truncation      : bool = True,
        ) -> Dict[str, Tensor]:
        if input_text2_batch is None:
            tokenize_res = self.tokenizer(
                input_text_batch,
                return_tensors = 'pt',
                padding        = padding,
                truncation     = truncation,
                max_length     = self.max_length
            ).to(self.device)
        else:
            tokenize_res = self.tokenizer(
                input_text_batch,
                input_text2_batch,
                return_tensors = 'pt',
                padding        = padding,
                truncation     = truncation,
                max_length     = self.max_length
            ).to(self.device)
        return tokenize_res

    def encode(self,
               tokenize_res: Dict[str, Tensor],
        ) -> Dict[str, Tensor]:
        hidden_states = self.encoder(
            **tokenize_res,
            return_dict          = True,
            output_hidden_states = True
        )
        return hidden_states

    @property
    def all_special_tokens(self):
        return self.tokenizer.all_special_tokens

    @property
    def all_special_ids(self):
        return self.tokenizer.all_special_ids

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def config(self):
        return self.encoder.config

    def convert_tokens_to_ids(self,
                              tokens: Union[str, List[str]]
        ) -> Union[int, List[int]]:
        return self.tokenizer.convert_tokens_to_ids(tokens)


class HFImageModel(HFModel):
    """
    Abstract class for all iamge models whose feature extractor and encoder are implemented
    with huggingface transformers.
    """

    def __init__(self,
                 model_dir : Union[str, os.PathLike],
                 feature_extractor : nn.Module,
                 encoder           : nn.Module
        ):
        super().__init__(model_dir)
        self.feature_extractor = feature_extractor
        self.encoder = encoder

    def save_model(self,
                   model_dir: Union[str, os.PathLike],
                   epoch_num: int = None,
                   step_num : int = None
        ) -> str:
        if model_dir.endswith('/'):
            model_dir = model_dir.strip('/')
        if epoch_num is not None:
            model_dir = model_dir + '-epoch-{}'.format(epoch_num)
        if step_num is not None:
            model_dir = model_dir + '-step-{}'.format(step_num)
        if os.path.exists(model_dir):
            pass
        else:
            os.mkdir(model_dir)
        self.feature_extractor.save_pretrained(model_dir)
        self.encoder.save_pretrained(model_dir)
        return model_dir

    def forward(self,
                images: Union[Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]],
                output_hidden_states: bool = None,
        ) -> Dict[str, Dict]:
        features = self.extract_feature(images)
        hidden_states = self.encode(features, output_hidden_states=output_hidden_states)
        output_dict = {
            'feature_res': features,
            'encode_res': hidden_states
        }
        return output_dict

    def extract_feature(self,
                        images: Union[Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]]
        ) -> Dict[str, Dict]:
        features = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
        return features

    def encode(self,
               features: Dict[str, Tensor],
               output_hidden_states: bool = None,
        ) -> Dict[str, Dict]:
        hidden_states = self.encoder(**features, output_hidden_states=output_hidden_states)
        return hidden_states




