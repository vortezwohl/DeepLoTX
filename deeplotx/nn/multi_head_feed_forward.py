from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.base_neural_network import BaseNeuralNetwork
from deeplotx.nn.feed_forward import FeedForward


class MultiHeadFeedForward(BaseNeuralNetwork):
    def __init__(self, feature_dim: int, num_heads: int = 1, num_layers: int = 1, expansion_factor: int | float = 2,
                 bias: bool = True, dropout_rate: float = 0.05, model_name: str | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__(in_features=feature_dim, out_features=feature_dim, model_name=model_name,
                         device=device, dtype=dtype)
        self._num_heads = num_heads
        self.expand_proj = nn.Linear(in_features=feature_dim, out_features=feature_dim * self._num_heads, bias=bias,
                                     device=self.device, dtype=self.dtype)
        self.ffn_heads = nn.ModuleList([FeedForward(feature_dim=feature_dim, num_layers=num_layers,
                                                    expansion_factor=expansion_factor, bias=bias,
                                                    dropout_rate=dropout_rate, device=self.device,
                                                    dtype=self.dtype) for _ in range(self._num_heads)])
        self.out_proj = nn.Linear(in_features=feature_dim * self._num_heads, out_features=feature_dim, bias=bias,
                                  device=self.device, dtype=self.dtype)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        x = self.expand_proj(x)
        x_heads = x.split(self.in_features, dim=-1)
        head_outs = [self.ffn_heads[_](x_heads[_]) for _ in range(self._num_heads)]
        return self.out_proj(torch.concat(head_outs, dim=-1))
