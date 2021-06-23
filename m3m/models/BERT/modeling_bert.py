# coding: utf-8
import os
import torch
import torch.nn as nn
from typing import List, Union, Optional
from ..model_base import HFModel
from argparse import Namespace
from transformers import BertTokenizer, BertModel


class BERT(HFModel):

    def __init__(self,
                 model_dir : Optional[Union[str, os.PathLike]],
                 max_length: int = 512
    ):
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        encoder   = BertModel.from_pretrained(model_dir)
        super(BERT, self).__init__(model_dir, tokenizer, encoder, max_length)




