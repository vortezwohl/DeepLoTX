from abc import abstractmethod

import torch
from torch import nn


class BaseTrainer(object):
    def __init__(self, model_name: str, num_epochs: int, learning_rate: float, batch_size: int, train_ratio: float):
        self._model_name = model_name
        self._num_epochs = num_epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._train_ratio = train_ratio

    @abstractmethod
    def train(self, *args, **kwargs): ...

    def save(self, model: nn.Module):
        torch.save(model.state_dict(), f'{self._model_name}.lotc.pth')
