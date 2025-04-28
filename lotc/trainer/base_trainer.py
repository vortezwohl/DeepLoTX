from abc import abstractmethod

from lotc.nn.base_neural_network import BaseNeuralNetwork


class BaseTrainer(object):
    def __init__(self, batch_size: int, train_ratio: float):
        self._batch_size = batch_size
        self._train_ratio = train_ratio
        self.model = None

    @abstractmethod
    def train(self, *args, **kwargs) -> BaseNeuralNetwork: ...
