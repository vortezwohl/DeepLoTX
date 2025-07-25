from typing_extensions import override

import torch

from deeplotx.nn.base_neural_network import BaseNeuralNetwork
from deeplotx.nn.feed_forward import FeedForward


class SelfAttention(BaseNeuralNetwork):
    def __init__(self, feature_dim: int, bias: bool = True, proj_layers: int = 1,
                 proj_expansion_factor: int | float = 1.25, dropout_rate: float = 0.02,
                 model_name: str | None = None, device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__(model_name=model_name, device=device, dtype=dtype)
        self._feature_dim = feature_dim
        self.q_proj = FeedForward(feature_dim=self._feature_dim, num_layers=proj_layers,
                                  expansion_factor=proj_expansion_factor,
                                  bias=bias, dropout_rate=dropout_rate, device=self.device, dtype=self.dtype)
        self.k_proj = FeedForward(feature_dim=self._feature_dim, num_layers=proj_layers,
                                  expansion_factor=proj_expansion_factor,
                                  bias=bias, dropout_rate=dropout_rate, device=self.device, dtype=self.dtype)
        self.v_proj = FeedForward(feature_dim=self._feature_dim, num_layers=proj_layers,
                                  expansion_factor=proj_expansion_factor,
                                  bias=bias, dropout_rate=dropout_rate, device=self.device, dtype=self.dtype)

    def _attention(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k = self.q_proj(x), self.k_proj(x)
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (self._feature_dim ** 0.5)
        attn = attn.masked_fill(mask == 0, -1e9) if mask is not None else attn
        return torch.softmax(attn, dim=-1)

    @override
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        if mask is not None:
            mask = self.ensure_device_and_dtype(mask, device=self.device, dtype=self.dtype)
        v = self.v_proj(x)
        return torch.matmul(self._attention(x, mask), v)
