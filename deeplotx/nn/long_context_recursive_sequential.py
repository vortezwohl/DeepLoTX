from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.recursive_sequential import RecursiveSequential
from deeplotx.nn.self_attention import SelfAttention


class LongContextRecursiveSequential(RecursiveSequential):
    def __init__(self, feature_dim: int, hidden_dim: int | None = None,
                 recursive_layers: int = 2, model_name: str | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__(input_dim=feature_dim, output_dim=feature_dim,
                         hidden_dim=hidden_dim, recursive_layers=recursive_layers,
                         model_name=model_name, device=device, dtype=dtype)
        self._feature_dim = feature_dim
        self.self_attention = SelfAttention(feature_dim=feature_dim)
        self.proj = nn.Linear(in_features=feature_dim * 2, out_features=feature_dim,
                              bias=True, device=self.device, dtype=self.dtype)

    @override
    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        x = torch.cat([self.self_attention(x), x], dim=-1)
        x = nn.LayerNorm(normalized_shape=x.shape[-1], eps=1e-9, device=self.device, dtype=self.dtype)(x)
        return super().forward(self.proj(x), state)
