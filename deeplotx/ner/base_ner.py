from deeplotx.ner.named_entity import NamedEntity


class BaseNER:
    def __init__(self): ...

    def extract_entities(self, s: str, *args, **kwargs) -> list[NamedEntity]: ...
