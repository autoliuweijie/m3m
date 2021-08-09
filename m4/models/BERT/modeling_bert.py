# coding: utf-8
import os
import torch
import torch.nn as nn
from typing import List, Union, Optional, Dict
from ..model_base import HFTextModel
from argparse import Namespace
from transformers import BertTokenizer, BertModel
from ...poolers import BertPooler
from ...targets import BertSentenceClassificationTarget
import logging


class BERT(HFTextModel):

    def __init__(self,
                 model_dir : Optional[Union[str, os.PathLike]],
                 max_length: int = 512
        ):
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        encoder   = BertModel.from_pretrained(model_dir)
        super(BERT, self).__init__(model_dir, tokenizer, encoder, max_length)


class BertForMatch(BERT):
    """
    Bert for sentence-pair classification.
    """

    def __init__(self,
                 model_dir: Optional[Union[str, os.PathLike]],
                 labels_num: int,
                 pool_type: str = 'cls',
                 max_length: int = 512
        ):
        super().__init__(model_dir, max_length)
        self.labels_num = labels_num
        self.pooler = BertPooler(pool_type)
        self.target = BertSentenceClassificationTarget(self.config.hidden_size, labels_num, self.config.hidden_dropout_prob)
        self.pooler_file_name = 'pooler.bin'
        self.target_file_name = 'target.bin'

        files = list(os.listdir(model_dir))
        if self.pooler_file_name in files:
            pooler_save_path = os.path.join(model_dir, self.pooler_file_name)
            pooler_state_dict = torch.load(pooler_save_path, map_location=torch.device('cpu'))
            self.pooler.load_state_dict(pooler_state_dict, strict=True)
        else:
            logging.warn(f'Weights of {self.__class__.__name__}.pooler are not trained!')

        if self.target_file_name in files:
            target_save_path = os.path.join(model_dir, self.target_file_name)
            target_state_dict = torch.load(target_save_path, map_location=torch.device('cpu'))
            self.target.load_state_dict(target_state_dict, strict=True)
        else:
            logging.warn(f'Weights of {self.__class__.__name__}.target are not trained!')

    def save_model(self,
                   model_dir: Union[str, os.PathLike],
                   epoch_num: int = None,
                   step_num : int = None
        ):
        model_dir = super().save_model(model_dir, epoch_num, step_num)
        pooler_save_path = os.path.join(model_dir, self.pooler_file_name)
        target_save_path = os.path.join(model_dir, self.target_file_name)

        pooler_state_dict = self.pooler.state_dict()
        target_state_dict = self.target.state_dict()

        torch.save(pooler_state_dict, pooler_save_path)
        torch.save(target_state_dict, target_save_path)
        return model_dir

    def forward(self,
                input_text_batch : List[str],
                input_text2_batch: List[str],
                padding          : bool = True,
                truncation       : bool = True,
                label_batch      : Union[List[int], torch.Tensor] = None  # batch_size
        ) -> Dict[str, torch.Tensor]:
        encode_res = super().forward(input_text_batch, input_text2_batch, padding, truncation)
        pool_res   = self.pooler(encode_res)

        if isinstance(label_batch, torch.Tensor):
            label_batch = label_batch.to(self.device)
        elif isinstance(label_batch, List):
            label_batch = torch.tensor(label_batch, device=self.device)
        else:
            pass
        target_res = self.target(pool_res, label_batch)

        probs = nn.functional.softmax(target_res['logits'], dim=1)

        output_dict  = {
            'probs': probs,
            'predict_labels': target_res['predict_labels'],
            'loss' : target_res.get('loss', None)
        }
        return output_dict


