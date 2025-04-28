from abc import abstractmethod

import torch


class BaseTrainer(object):
    def __init__(self, model_name: str, num_epochs: int,
                 learning_rate: float, batch_size: int, train_ratio: float,
                 train_loss_threshold: float, valid_loss_threshold: float):
        self._model_name = model_name
        self._num_epochs = num_epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._train_ratio = train_ratio
        self._train_loss_threshold = train_loss_threshold
        self._valid_loss_threshold = valid_loss_threshold
        self.model = None

    @abstractmethod
    def train(self, *args, **kwargs): ...

    def save(self):
        torch.save(self.model.state_dict(), f'{self._model_name}.lotc.pth')
        return self.model

    def load(self):
        self.model.load_state_dict(torch.load(f'{self._model_name}.lotc.pth'))
        return self.model
