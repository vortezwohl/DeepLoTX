from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.base_neural_network import BaseNeuralNetwork
from deeplotx.nn.feed_forward import FeedForward
from deeplotx.nn.attention import Attention


class RoFormerEncoder(BaseNeuralNetwork):
    def __init__(self, feature_dim: int, bias: bool = True,
                 ffn_layers: int = 1, ffn_expansion_factor: int | float = 2,
                 dropout_rate: float = 0.02, model_name: str | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(in_features=feature_dim, out_features=feature_dim,
                         model_name=model_name, device=device, dtype=dtype)
        self.self_attention = Attention(feature_dim=feature_dim, bias=bias, positional=True,
                                        proj_layers=kwargs.get('attn_ffn_layers', 1),
                                        proj_expansion_factor=kwargs.get('attn_expansion_factor', ffn_expansion_factor),
                                        dropout_rate=kwargs.get('attn_dropout_rate', dropout_rate),
                                        device=self.device, dtype=self.dtype, **kwargs)
        self.ffn = FeedForward(feature_dim=feature_dim * 2, num_layers=ffn_layers,
                               expansion_factor=ffn_expansion_factor,
                               bias=bias, dropout_rate=dropout_rate,
                               device=self.device, dtype=self.dtype)
        self.layer_norm = nn.LayerNorm(normalized_shape=feature_dim, eps=1e-9,
                                       device=self.device, dtype=self.dtype)
        self.__proj = nn.Linear(in_features=feature_dim * 2, out_features=feature_dim,
                                bias=bias, device=self.device, dtype=self.dtype)

    @override
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        if mask is not None:
            mask = self.ensure_device_and_dtype(mask, device=self.device, dtype=self.dtype)
        attn = self.self_attention(x=self.layer_norm(x), y=None, mask=mask)
        x = torch.concat([attn, x], dim=-1)
        return self.__proj(self.ffn(x))
