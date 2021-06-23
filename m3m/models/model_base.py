# coding: utf-8
import os
import shutil
import torch
import torch.nn as nn
from typing import Union, List, Dict


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

    def save_model(self,
                   model_dir: Union[str, os.PathLike]
    ):
        raise NotImplementedError

    def forward(self,
                *args,
                **kwargs,
    ):
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

    def __init__(self,
                 model_dir : Union[str, os.PathLike],
                 tokenizer : nn.Module,
                 encoder   : nn.Module,
                 max_length: int
    ):
        super().__init__(model_dir, tokenizer, encoder)
        self.max_length = max_length


    def save_model(self,
                   model_dir: Union[str, os.PathLike]
    ):
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        self.encoder.save_pretrained(model_dir)
        return model_dir

    def forward(self,
                input_text_batch: List[str],
                padding    = True,
                truncation = True,
    ):
        return self.encode(input_text_batch, padding, truncation, self.max_length)

    def encode(self,
               input_text_batch: List[str],
               padding: bool,
               truncation: bool,
               max_length: int,
    ) -> Dict[str, Dict]:
        tokenize_res = self.tokenizer(
            input_text_batch,
            return_tensors = 'pt',
            padding        = padding,
            truncation     = truncation,
            max_length     = max_length
        )
        hidden_states = self.encoder(
            **tokenize_res,
            return_dict          = True,
            output_hidden_states = True
        )
        output_dict = {
            'tokenize_res': tokenize_res,
            'encode_res'  : hidden_states
        }
        return output_dict

