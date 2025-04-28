from typing_extensions import override

import torch

from lotc.nn.linear_regression import LinearRegression


class LogisticRegression(LinearRegression):
    def __init__(self, input_dim: int, train: bool = False):
        super().__init__(input_dim=input_dim, output_dim=1, train=train)

    @override
    def forward(self, x) -> torch.Tensor:
        return torch.sigmoid(super().forward(x))
