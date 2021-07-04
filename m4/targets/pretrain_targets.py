# coding: utf-8
"""
Some targets for pretraining models.

@author: weijie liu
"""
import torch
import torch.nn as nn
from ..utils.activations import ACT2FN
from typing import Tuple, Dict


class MLMTarget(nn.Module):
    """
    Masked Languege Model.
    """

    def __init__(self,
                 hidden_size: int,
                 vocab_size : int,
                 active_func: str = 'gelu',
                 layer_norm_eps: float = 1e-12,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.active_func = ACT2FN[active_func]
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self,
                hidden_states: torch.Tensor,       # batch_size x seq_len x hidden_size
                labels       : torch.Tensor = None # batch_size x seq_len
        ) -> Dict[str, torch.Tensor]:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.active_func(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        predict_ids = logits.argmax(dim = -1)  # batch_size x seq_len

        mlm_loss = None
        if isinstance(labels, torch.Tensor):
            # only compute loss for masked tokens
            ped_logits = logits[labels >= 0, :]
            tgt_labels = labels[labels >= 0]
            mlm_loss   = self.loss_fct(ped_logits.view(-1, self.vocab_size), tgt_labels.view(-1))

            ped_mask_ids = predict_ids[labels >= 0]
            correct_num  = torch.sum(ped_mask_ids.eq(tgt_labels).float())
            total_num    = len(ped_mask_ids.view(-1))
            accuracy     = correct_num / total_num

            ret_dict = {
                'logits'  : logits,
                'predict_ids': predict_ids,
                'mlm_loss': mlm_loss,
                'correct_num': correct_num,
                'total_num': total_num,
                'accuracy': accuracy
            }
        else:
            ret_dict = {
                'logits'  : logits,
                'predict_ids': predict_ids,
            }

        return ret_dict






