from abc import abstractmethod

from deeplotx.similarity.base_similarity import BaseSimilarity


class BaseStringSimilarity(BaseSimilarity):
    def __init__(self, ignore_case: bool, *args, **kwargs):
        super().__init__()
        self._ignore_case = ignore_case

    @abstractmethod
    def __call__(self, *args, **kwargs): ...

    @abstractmethod
    def rank(self, s: str, S: list[str]) -> list[str]: ...
