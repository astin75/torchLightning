from torchvision import models
from torch import nn


def resnet50(dims=256):
    resnet = models.resnet50(pretrained=True)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, dims)
    return resnet

