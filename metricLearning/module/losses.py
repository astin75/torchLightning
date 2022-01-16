import torch
from torch import nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineEmbeddingLoss(margin=0.5, reduction='mean')

    def forward(self, x):
        vector_1, vector_2, mask = x
        mask = torch.where(mask,
                           torch.ones_like(mask, dtype=torch.float),
                           -torch.ones_like(mask, dtype=torch.float))
        loss = self.cos(vector_1, vector_2, mask)
        return loss