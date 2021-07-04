# coding: utf-8
import torch
import torch.nn as nn
from torch import Tensor
from ..utils.similarity import cosine_sim


class BaseBatchPostivePairLoss(nn.Module):
    """
    For a pair of vectors (vecs_a[i], vecs_b[j]), we assume it as a postive
    sample if i==j, otherwise as a negative sample.
    """

    def __init__(self,
                 scale: float,
                 similarity_fct = cosine_sim
    ):
        super().__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct

    def forward(self,
                vecs_a: Tensor,  # batch_size * emb_size
                vecs_b: Tensor   # batch_size * emb_size
    ) -> Tensor:
        raise NotImplementedError


class BatchPostivePairXentLoss(BaseBatchPostivePairLoss):

    def __init__(self,
                 scale = 1.0,
                 similarity_fct = cosine_sim
    ):
        super().__init__(scale, similarity_fct)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self,
                vecs_a: Tensor,  # batch_size * emb_size
                vecs_b: Tensor   # batch_size * emb_size
    ) -> Tensor:
        scores = self.similarity_fct(vecs_a, vecs_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        return self.cross_entropy_loss(scores, labels)

