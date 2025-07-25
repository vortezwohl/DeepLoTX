from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.recursive_sequential import RecursiveSequential
from deeplotx.nn.self_attention import SelfAttention


class LongContextRecursiveSequential(RecursiveSequential):
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int | None = None, recursive_layers: int = 2,
                 ffn_layers: int = 1, ffn_expansion_factor: int | float = 2,
                 ffn_bias: bool = True, ffn_dropout_rate: float = 0.05,
                 model_name: str | None = None, device: str | None = None,
                 dtype: torch.dtype | None = None, **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim,
                         hidden_dim=hidden_dim, recursive_layers=recursive_layers,
                         ffn_layers=ffn_layers, ffn_expansion_factor=ffn_expansion_factor,
                         ffn_bias=ffn_bias, ffn_dropout_rate=ffn_dropout_rate,
                         model_name=model_name, device=device, dtype=dtype)
        self.self_attention = SelfAttention(feature_dim=input_dim, bias=kwargs.get('attn_proj_bias', ffn_bias),
                                            proj_layers=kwargs.get('attn_proj_layers', 1),
                                            proj_expansion_factor=kwargs.get('attn_proj_expansion_factor', ffn_expansion_factor),
                                            dropout_rate=kwargs.get('attn_proj_dropout_rate', ffn_dropout_rate))
        self.__proj = nn.Linear(in_features=input_dim * 2, out_features=input_dim,
                                bias=ffn_bias, device=self.device, dtype=self.dtype)

    @override
    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        x = torch.cat([self.self_attention(x), x], dim=-1)
        x = nn.LayerNorm(normalized_shape=x.shape[-1], eps=1e-9, device=self.device, dtype=self.dtype)(x)
        return super().forward(self.__proj(x), state)
