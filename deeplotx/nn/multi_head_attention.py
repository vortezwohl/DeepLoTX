from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.base_neural_network import BaseNeuralNetwork
from deeplotx.nn.attention import Attention


class MultiHeadAttention(BaseNeuralNetwork):
    def __init__(self, feature_dim: int, num_heads: int = 1, bias: bool = True, positional: bool = True,
                 proj_layers: int = 1, proj_expansion_factor: int | float = 1.5, dropout_rate: float = 0.02,
                 model_name: str | None = None, device: str | None = None, dtype: torch.dtype | None = None,
                 **kwargs):
        super().__init__(in_features=feature_dim, out_features=feature_dim, model_name=model_name,
                         device=device, dtype=dtype)
        self._num_heads = num_heads
        self.expand_proj = nn.Linear(in_features=feature_dim, out_features=feature_dim * self._num_heads, bias=bias,
                                     device=self.device, dtype=self.dtype)
        self.attn_heads = nn.ModuleList([Attention(feature_dim=feature_dim, bias=bias, positional=positional,
                                                   proj_layers=proj_layers, proj_expansion_factor=proj_expansion_factor,
                                                   dropout_rate=dropout_rate, device=self.device, dtype=self.dtype,
                                                   **kwargs) for _ in range(self._num_heads)])
        self.out_proj = nn.Linear(in_features=feature_dim * self._num_heads, out_features=feature_dim, bias=bias,
                                  device=self.device, dtype=self.dtype)

    @override
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        y = x if y is None else self.ensure_device_and_dtype(y, device=self.device, dtype=self.dtype)
        x, y = self.expand_proj(x), self.expand_proj(y)
        x_heads, y_heads = x.split(self.in_features, dim=-1), y.split(self.in_features, dim=-1)
        head_outs = [self.attn_heads[_](x=x_heads[_], y=y_heads[_], mask=mask) for _ in range(self._num_heads)]
        return self.out_proj(torch.concat(head_outs, dim=-1))
