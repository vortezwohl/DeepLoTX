import torch
from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, train: bool = False):
        super().__init__()
        self._train = train
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc1_to_fc4_res = nn.Linear(512, 32)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, output_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.05)

    def forward(self, x) -> torch.Tensor:
        fc1_out = self.leaky_relu(self.fc1(x))
        x = nn.LayerNorm(normalized_shape=512, eps=1e-9)(fc1_out)
        x = torch.dropout(x, p=0.2, train=self._train)
        x = self.leaky_relu(self.fc2(x))
        x = nn.LayerNorm(normalized_shape=128, eps=1e-9)(x)
        x = torch.dropout(x, p=0.2, train=self._train)
        x = self.leaky_relu(self.fc3(x))
        x = torch.dropout(x, p=0.2, train=self._train)
        x = self.leaky_relu(self.fc4(x)) + self.fc1_to_fc4_res(fc1_out)
        x = self.fc5(x)
        return x

    def predict(self, x) -> torch.Tensor:
        __train = self._train
        self._train = False
        with torch.no_grad():
            res = self.forward(x)
        self._train = __train
        return res
