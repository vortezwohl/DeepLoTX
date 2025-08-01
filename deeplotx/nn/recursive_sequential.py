from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.base_neural_network import BaseNeuralNetwork
from deeplotx.nn.linear_regression import LinearRegression


class RecursiveSequential(BaseNeuralNetwork):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True,
                 recursive_layers: int = 1, recursive_hidden_dim: int | None = None,
                 ffn_layers: int = 1, ffn_expansion_factor: int | float = 2, dropout_rate: float = 0.05,
                 model_name: str | None = None, device: str | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(in_features=input_dim, out_features=output_dim, model_name=model_name,
                         device=device, dtype=dtype)
        if recursive_hidden_dim is None:
            recursive_hidden_dim = input_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=recursive_hidden_dim,
                            num_layers=recursive_layers, batch_first=True,
                            bias=True, bidirectional=True, device=self.device,
                            dtype=self.dtype)
        self.out_proj = LinearRegression(input_dim=recursive_hidden_dim * 2, output_dim=output_dim,
                                         num_heads=kwargs.get('ffn_heads', 1), head_layers=kwargs.get('ffn_head_layers', 1),
                                         num_layers=ffn_layers, expansion_factor=ffn_expansion_factor,
                                         bias=bias, dropout_rate=dropout_rate, device=self.device, dtype=self.dtype)

    def initial_state(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size, device=self.device, dtype=self.dtype)
        return zeros, zeros

    @override
    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        state = (self.ensure_device_and_dtype(state[0], device=self.device, dtype=self.dtype),
                 self.ensure_device_and_dtype(state[1], device=self.device, dtype=self.dtype))
        x, (hidden_state, cell_state) = self.lstm(x, state)
        x = x[:, -1, :]
        x = self.out_proj(x)
        return x, (hidden_state, cell_state)

    @override
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        training_state_dict = dict()
        for m in self.modules():
            training_state_dict[m] = m.training
            m.training = False
        with torch.no_grad():
            res = self.forward(x.unsqueeze(0), self.initial_state(batch_size=1))[0]
        for m, training_state in training_state_dict.items():
            m.training = training_state
        return res
