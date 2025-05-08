from typing_extensions import override

import torch

from deeplotx.nn.linear_regression import LinearRegression


class LogisticRegression(LinearRegression):
    def __init__(self, input_dim: int, output_dim: int = 1, model_name: str | None = None):
        super().__init__(input_dim=input_dim, output_dim=output_dim, model_name=model_name)

    @override
    def forward(self, x) -> torch.Tensor:
        return torch.sigmoid(super().forward(x))
