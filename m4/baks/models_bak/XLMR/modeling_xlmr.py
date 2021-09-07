# coding: utf-8
import os
import torch
import torch.nn as nn
from typing import List, Union, Optional, Dict
from ..model_base import HFTextModel
from transformers import XLMRobertaTokenizer, XLMRobertaModel


class XLMR(HFTextModel):

    def __init__(self,
                 model_dir : Optional[Union[str, os.PathLike]],
                 max_length: int = 512
        ):
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_dir)
        encoder   = XLMRobertaModel.from_pretrained(model_dir)
        super(XLMR, self).__init__(model_dir, tokenizer, encoder, max_length)

