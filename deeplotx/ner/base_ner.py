from deeplotx.ner.named_entity import NamedEntity, NamedPerson


class BaseNER:
    def __init__(self): ...

    def __call__(self, s: str, *args, **kwargs) -> list[NamedEntity | NamedPerson]: ...
