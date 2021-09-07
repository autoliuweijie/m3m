# coding: utf-8
"""
Some targets for bert-like models.

@author: weijie liu
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict


class BertSentenceClassificationTarget(nn.Module):

    def __init__(self,
                 hidden_size : int,
                 labels_num  : int,
                 dropout_prob: float=0.15
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.labels_num   = labels_num
        self.dense1     = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout    = nn.Dropout(dropout_prob)
        self.dense2     = nn.Linear(hidden_size, labels_num)
        self.loss_fct   = nn.CrossEntropyLoss()

    def forward(self,
                hidden_state: torch.Tensor, # batch_size x hidden_size
                labels      : torch.Tensor = None  # batch_size
        ) -> Dict[str, torch.Tensor]:
        hidden_state = self.dense1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state)
        logits       = self.dense2(hidden_state)  # batch_size x labels_num
        predict_labels = torch.argmax(logits, dim=1)

        ret_dict = {
            'logits': logits,
            'predict_labels': predict_labels,
        }

        if labels is not None:
           loss_value = self.loss_fct(logits.view(-1, self.labels_num), labels.view(-1))
           ret_dict['loss'] = loss_value

        return ret_dict
