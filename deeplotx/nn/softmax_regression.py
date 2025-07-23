from typing_extensions import override

import torch

from deeplotx.nn.linear_regression import LinearRegression


class SoftmaxRegression(LinearRegression):
    def __init__(self, input_dim: int, output_dim: int, model_name: str | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__(input_dim=input_dim, output_dim=output_dim, model_name=model_name, device=device, dtype=dtype)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        return torch.softmax(super().forward(x), dim=-1, dtype=self.dtype)
