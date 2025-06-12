from abc import abstractmethod

import torch
from torch import nn


class BaseNeuralNetwork(nn.Module):
    def __init__(self, model_name: str | None = None, device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self._model_name = model_name \
            if model_name is not None \
            else self.__class__.__name__
        self.device = torch.device(device) if device is not None \
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype is not None else torch.float32

    @staticmethod
    def ensure_device_and_dtype(x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if x.device != device:
            x = x.to(device)
        if x.dtype != dtype:
            x = x.to(dtype)
        return x

    def l1(self, _lambda: float = 1e-4) -> torch.Tensor:
        def _l1() -> torch.Tensor:
            l2_reg = torch.tensor(0., device=self.device, dtype=self.dtype)
            for param in self.parameters():
                l2_reg += (torch.abs(param)).sum()
            return l2_reg
        return _lambda * _l1()

    def l2(self, _lambda: float = 1e-4) -> torch.Tensor:
        def _l2() -> torch.Tensor:
            l2_reg = torch.tensor(0., device=self.device, dtype=self.dtype)
            for param in self.parameters():
                l2_reg += (torch.pow(param, exponent=2.)).sum()
            return l2_reg
        return _lambda * _l2() / 2.

    def elastic_net(self, alpha: float = 1e-4, rho: float = 0.5) -> torch.Tensor:
        return alpha * (rho * self.l1(_lambda=1.) + (1 - rho) * self.l2(_lambda=1.))

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor: ...

    def predict(self, x) -> torch.Tensor:
        x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
        __train = self.training
        self.training = False
        with torch.no_grad():
            res = self.forward(x)
        self.training = __train
        return res

    def save(self):
        torch.save(self.state_dict(), f'{self._model_name}.deeplotx')
        return self

    def load(self):
        self.load_state_dict(torch.load(f'{self._model_name}.deeplotx', map_location=self.device, weights_only=True))
        return self
