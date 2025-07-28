from typing_extensions import override

import torch

from deeplotx.nn.base_neural_network import BaseNeuralNetwork


class RoPE(BaseNeuralNetwork):
    def __init__(self, feature_dim: int, base: int = 10000, device: str | None = None, dtype: torch.dtype = torch.float32):
        super().__init__(in_features=feature_dim, out_features=feature_dim, model_name=None,
                         device=device, dtype=dtype)
        assert feature_dim % 2 == 0, f'feature_dim must be divisible by 2.'
        self._base = base
        self._num_groups = feature_dim // 2
        self._inv_freq = 1.0 / (base ** (torch.arange(start=0, end=self._num_groups, step=1).float() / self._num_groups))
        self.register_buffer('inv_freq', self._inv_freq)

    @property
    def dim(self):
        return self._dim

    @property
    def base(self):
        return self._base

    def rotate_half(self, _t: torch.Tensor) -> torch.Tensor:
        return torch.cat((- _t[..., self._num_groups:], _t[..., :self._num_groups]), dim=-1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        *other_dims, seq_len, feature_dim = x.shape
        assert feature_dim == self.in_features, f"feature_dim of x doesn't match with defined feature_dim {self.in_features}."
        t = torch.arange(start=0, end=seq_len, step=1, device=self.device, dtype=self.dtype)
        freq = torch.outer(t, self._inv_freq)
        emb = torch.cat((freq, freq), dim=-1)
        sin_emb, cos_emb = emb.sin(), emb.cos()
        return x * cos_emb + self.rotate_half(x) * sin_emb
