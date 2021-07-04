# coding: utf-8
import os
import shutil
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, List, Dict
import time


class BaseModel(nn.Module):
    """
    Abstract Class for all models.
    """

    def __init__(self,
                 model_dir: Union[str, os.PathLike],
                 tokenizer: nn.Module,
                 encoder  : nn.Module,
        ):
        super().__init__()
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        self.encoder   = encoder
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

    def __init__(self,
                 model_dir : Union[str, os.PathLike],
                 tokenizer : nn.Module,
                 encoder   : nn.Module,
                 max_length: int
        ):
        super().__init__(model_dir, tokenizer, encoder)
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


