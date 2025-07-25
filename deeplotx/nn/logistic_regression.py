from typing_extensions import override

import torch

from deeplotx.nn.linear_regression import LinearRegression


class LogisticRegression(LinearRegression):
    def __init__(self, input_dim: int, output_dim: int = 1, num_layers: int = 1, expansion_factor: int | float = 1.25,
                 bias: bool = True, dropout_rate: float = 0.1, model_name: str | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__(input_dim=input_dim, output_dim=output_dim, num_layers=num_layers,
                         expansion_factor=expansion_factor, bias=bias, dropout_rate=dropout_rate,
                         model_name=model_name, device=device, dtype=dtype)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        return torch.sigmoid(super().forward(x))
