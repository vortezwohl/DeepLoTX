from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.attention import DEFAULT_THETA
from deeplotx.nn.recursive_sequential import RecursiveSequential
from deeplotx.nn.roformer_encoder import RoFormerEncoder


class LongContextRecursiveSequential(RecursiveSequential):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True,
                 encoder_layers: int = 1, attn_heads: int = 1, recursive_layers: int = 2, recursive_hidden_dim: int | None = None,
                 ffn_layers: int = 1, ffn_expansion_factor: int | float = 2, dropout_rate: float = 0.05,
                 model_name: str | None = None, device: str | None = None, dtype: torch.dtype | None = None,
                 **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim, bias=bias,
                         recursive_layers=recursive_layers, recursive_hidden_dim=recursive_hidden_dim,
                         ffn_layers=ffn_layers, ffn_expansion_factor=ffn_expansion_factor, dropout_rate=dropout_rate,
                         model_name=model_name, device=device, dtype=dtype)
        self.roformer_encoders = nn.ModuleList([RoFormerEncoder(feature_dim=input_dim, attn_heads=attn_heads, bias=bias,
                                                                ffn_layers=kwargs.get('encoder_ffn_layers', ffn_layers),
                                                                ffn_expansion_factor=kwargs.get('encoder_ffn_expansion_factor', ffn_expansion_factor),
                                                                dropout_rate=kwargs.get('encoder_dropout_rate', dropout_rate),
                                                                attn_ffn_layers=kwargs.get('attn_ffn_layers', 1),
                                                                attn_expansion_factor=kwargs.get('attn_expansion_factor', ffn_expansion_factor),
                                                                attn_dropout_rate=kwargs.get('attn_dropout_rate', dropout_rate),
                                                                theta=kwargs.get('theta', DEFAULT_THETA),
                                                                device=self.device, dtype=self.dtype) for _ in range(encoder_layers)])

    @override
    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        for roformer_encoder in self.roformer_encoders:
            x = roformer_encoder(x)
        return super().forward(x, state)
