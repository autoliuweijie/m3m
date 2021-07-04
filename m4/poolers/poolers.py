# coding: utf-8
import torch
import torch.nn as nn
from typing import Dict


class BasePooler(nn.Module):
    """
    Abstract class for all Poolers.
    """

    def __init__(self,
                 *args,
                 **kwargs
    ):
        super().__init__()

    def forward(self,
                encode_output: Dict
    ):
        raise NotImplementedError


class BertPooler(BasePooler):
    """
    Pooler for BERT-like models.
    """

    def __init__(self,
                 pool_type: str
    ):
        super().__init__()
        assert pool_type in ["cls", "avg", "avg_top2", "avg_first_last"], \
            "unrecognized pooling type %s" % self.pooler_type
        self.pool_type = pool_type

    def forward(self,
                encode_output: Dict
    ):
        attention_mask = encode_output['tokenize_res'].attention_mask
        last_hidden    = encode_output['encode_res'].last_hidden_state
        hidden_states  = encode_output['encode_res'].hidden_states

        if self.pool_type == 'cls':
            return last_hidden[:, 0]
        elif self.pool_type == 'avg':
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pool_type == 'ave_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden   = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pool_type == 'avg_first_last':
            first_hidden  = hidden_states[0]
            last_hidden   = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result

