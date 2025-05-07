from typing_extensions import override

import torch
from torch import nn

from lotc.nn.base_neural_network import BaseNeuralNetwork


class LinearRegression(BaseNeuralNetwork):
    def __init__(self, input_dim: int, output_dim: int, model_name: str | None = None):
        super().__init__(model_name=model_name)
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc1_to_fc4_res = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(1024, 768)
        self.fc3 = nn.Linear(768, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)
        self.parametric_relu_1 = nn.PReLU(num_parameters=1, init=5e-3)
        self.parametric_relu_2 = nn.PReLU(num_parameters=1, init=5e-3)
        self.parametric_relu_3 = nn.PReLU(num_parameters=1, init=5e-3)
        self.parametric_relu_4 = nn.PReLU(num_parameters=1, init=5e-3)

    @override
    def forward(self, x) -> torch.Tensor:
        fc1_out = self.parametric_relu_1(self.fc1(x))
        x = nn.LayerNorm(normalized_shape=1024, eps=1e-9)(fc1_out)
        x = torch.dropout(x, p=0.2, train=self.training)
        x = self.parametric_relu_2(self.fc2(x))
        x = nn.LayerNorm(normalized_shape=768, eps=1e-9)(x)
        x = torch.dropout(x, p=0.2, train=self.training)
        x = self.parametric_relu_3(self.fc3(x))
        x = torch.dropout(x, p=0.2, train=self.training)
        x = self.parametric_relu_4(self.fc4(x)) + self.fc1_to_fc4_res(fc1_out)
        x = self.fc5(x)
        return x
