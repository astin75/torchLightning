from torch import nn
import torch


def cosine_sim(x1, x2, dim=1, eps=1e-8):

    inner_product = torch.mm(x1, x2.t())
    w1 = torch.linalg.norm(x1, 2, dim)
    w2 = torch.linalg.norm(x2, 2, dim)
    return inner_product/torch.outer(w1, w2).clamp(min=eps)


class MarginCosineProudct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        feature, label = x
        cos = cosine_sim(feature, self.weight)

        # convert to one-hot
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = self.s * (cos - one_hot*self.m)
        return output