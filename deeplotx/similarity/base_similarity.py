from abc import abstractmethod

from typing_extensions import Callable, Any


class BaseSimilarity(Callable):
    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def __call__(self, *args, **kwargs): ...

    @abstractmethod
    def rank(self, s: Any, S: list[Any]) -> list[Any]: ...
