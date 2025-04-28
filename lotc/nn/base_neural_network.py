import torch
from torch import nn


class BaseNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x) -> torch.Tensor: ...

    def predict(self, x) -> torch.Tensor:
        __train = self.training
        self.training = False
        with torch.no_grad():
            res = self.forward(x)
        self.training = __train
        return res
