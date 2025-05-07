import torch
from torch import nn


class BaseNeuralNetwork(nn.Module):
    def __init__(self, model_name: str | None = None):
        super().__init__()
        self._model_name = model_name \
            if model_name is not None \
            else self.__class__.__name__

    def l1(self, _lambda: float = 0.) -> torch.Tensor:
        def _l1() -> torch.Tensor:
            l2_reg = torch.tensor(0.)
            for param in self.parameters():
                l2_reg += (torch.abs(param)).sum()
            return l2_reg
        return _lambda * _l1()

    def l2(self, _lambda: float = 0.) -> torch.Tensor:
        def _l2() -> torch.Tensor:
            l2_reg = torch.tensor(0.)
            for param in self.parameters():
                l2_reg += (torch.pow(param, exponent=2.)).sum()
            return l2_reg
        return _lambda * _l2()

    def forward(self, x) -> torch.Tensor: ...

    def predict(self, x) -> torch.Tensor:
        __train = self.training
        self.training = False
        with torch.no_grad():
            res = self.forward(x)
        self.training = __train
        return res

    def save(self):
        torch.save(self.state_dict(), f'{self._model_name}.lotc.pth')
        return self

    def load(self):
        self.load_state_dict(torch.load(f'{self._model_name}.lotc.pth'))
        return self
