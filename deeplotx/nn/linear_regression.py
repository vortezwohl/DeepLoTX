from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.base_neural_network import BaseNeuralNetwork
from deeplotx.nn.multi_head_feed_forward import MultiHeadFeedForward


class LinearRegression(BaseNeuralNetwork):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 1, num_layers: int = 1,
                 expansion_factor: int | float = 1.5, bias: bool = True, dropout_rate: float = 0.1,
                 model_name: str | None = None, device: str | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(in_features=input_dim, out_features=output_dim, model_name=model_name, device=device, dtype=dtype)
        self.multi_head_ffn_layers = nn.ModuleList([MultiHeadFeedForward(feature_dim=input_dim, num_heads=num_heads,
                                                                         num_layers=kwargs.get('head_layers', 1),
                                                                         expansion_factor=expansion_factor,
                                                                         bias=bias, dropout_rate=dropout_rate,
                                                                         device=self.device, dtype=self.dtype) for _ in range(num_layers)])
        self.proj = nn.Linear(in_features=input_dim, out_features=output_dim,
                              bias=bias, device=self.device, dtype=self.dtype)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        residual = x
        for ffn in self.multi_head_ffn_layers:
            x = ffn(x)
        return self.proj(x + residual)
