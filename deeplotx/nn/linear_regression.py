from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.base_neural_network import BaseNeuralNetwork
from deeplotx.nn.feed_forward import FeedForward


class LinearRegression(BaseNeuralNetwork):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1,
                 expansion_factor: int | float = 1.25, bias: bool = True, model_name: str | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__(model_name=model_name, device=device, dtype=dtype)
        self.ffn = FeedForward(feature_dim=input_dim, num_layers=num_layers, expansion_factor=expansion_factor,
                               bias=bias, device=self.device, dtype=self.dtype)
        self.proj = nn.Linear(in_features=input_dim, out_features=output_dim, bias=bias, device=self.device, dtype=self.dtype)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        residual = x
        x = self.ffn(x) + residual
        return self.proj(x)
