from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.base_neural_network import BaseNeuralNetwork


class FeedForwardUnit(BaseNeuralNetwork):
    def __init__(self, feature_dim: int, expansion_factor: int | float = 2,
                 bias: bool = True, dropout_rate: float = 0.05, model_name: str | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__(in_features=feature_dim, out_features=feature_dim, model_name=model_name, device=device, dtype=dtype)
        self._dropout_rate = dropout_rate
        self.fc1 = nn.Linear(feature_dim, int(feature_dim * expansion_factor), bias=bias,
                             device=self.device, dtype=self.dtype)
        self.fc2 = nn.Linear(int(feature_dim * expansion_factor), feature_dim, bias=bias,
                             device=self.device, dtype=self.dtype)
        self.parametric_relu_1 = nn.PReLU(num_parameters=1, init=5e-3,
                                          device=self.device, dtype=self.dtype)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.fc1.in_features, eps=1e-9,
                                       device=self.device, dtype=self.dtype)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        residual = x
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.parametric_relu_1(x)
        if self._dropout_rate > .0:
            x = torch.dropout(x, p=self._dropout_rate, train=self.training)
        return self.fc2(x) + residual


class FeedForward(BaseNeuralNetwork):
    def __init__(self, feature_dim: int, num_layers: int = 1, expansion_factor: int | float = 2,
                 bias: bool = True, dropout_rate: float = 0.05, model_name: str | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None):
        if num_layers < 1:
            raise ValueError('num_layers cannot be less than 1.')
        super().__init__(in_features=feature_dim, out_features=feature_dim, model_name=model_name, device=device, dtype=dtype)
        self.ffn_layers = nn.ModuleList([FeedForwardUnit(feature_dim=feature_dim,
                                                         expansion_factor=expansion_factor, bias=bias,
                                                         dropout_rate=dropout_rate,
                                                         device=self.device, dtype=self.dtype)] * num_layers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        for ffn in self.ffn_layers:
            x = ffn(x)
        return x
