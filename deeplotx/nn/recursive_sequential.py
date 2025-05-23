from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.base_neural_network import BaseNeuralNetwork
from deeplotx.nn import LinearRegression


class RecursiveSequential(BaseNeuralNetwork):
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int | None = None, recursive_layers: int = 2,
                 model_name: str | None = None):
        super().__init__(model_name=model_name)
        if hidden_dim is None:
            hidden_dim = input_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=recursive_layers, batch_first=True,
                            bias=True, bidirectional=True)
        self.regressive_head = LinearRegression(input_dim=hidden_dim * 2, output_dim=output_dim)

    def initial_state(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size)
        return zeros, zeros

    @override
    def forward(self, x, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x, (hidden_state, cell_state) = self.lstm(x, state)
        x = self.regressive_head(x[:, -1, :])
        return x, (hidden_state, cell_state)
