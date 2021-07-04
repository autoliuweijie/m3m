# coding: utf-8
"""
Some utils for pretraining models.

@author: weijie liu
"""
import torch
import torch.nn as nn
from typing import Tuple, Union



def get_special_token_mask(
        input_ids        : torch.Tensor,               # batch_size x seq_len
        special_token_ids: Union[list, set] = set([])  # a list of special token ids
    ) -> torch.Tensor:  # batch_size x seq_len
    special_token_mask = []
    for line in input_ids.tolist():
        line_mask = [1 if tid in special_token_ids else 0 for tid in line]
        special_token_mask.append(line_mask)
    special_token_mask = torch.tensor(special_token_mask, dtype=torch.bool, device=input_ids.device)
    return special_token_mask


def mask_token_ids(
        input_ids        : torch.Tensor,  # batch_size x seq_len
        mask_id          : int,
        vocab_size       : int,
        mlm_prob         : float = 0.15,  # probability for MLM
        mask_prob        : float = 0.8,   # probability for [MASK]
        special_token_ids: Union[list, set] = set([])      # a list of special token ids
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask token_id for MLM pretraining.

    @ref: https://github.com/huggingface/transformers/src/transformers/data/data_collator.py#L396
    """
    device = input_ids.device
    input_ids = input_ids.clone()

    # Select mask ids with [mlm_prob]
    labels = input_ids.clone()
    special_token_mask = get_special_token_mask(input_ids, special_token_ids)
    probability_matrix = torch.full(labels.shape, mlm_prob, device=device)
    probability_matrix.masked_fill_(special_token_mask, value=0.0) # batch_size x seq_len
    masked_indices = torch.bernoulli(probability_matrix).bool()     # 伯努利分布
    labels[~masked_indices] = -100  # labels >= 0 are masked, we only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = mask_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    return input_ids, labels


def whole_word_mask_token_ids(
        input_ids  : torch.Tensor,  # batch_size x seq_len
        mask_id    : int,
        mlm_prob   : float = 0.15,  # probability for MLM
        mask_prob  : float = 0.8,   # 80% MASK
        random_prob: float = 0.1,   # 10% random
        origin_prob: float = 0.1,   # 10% origin
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    pass


