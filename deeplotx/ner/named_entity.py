from dataclasses import dataclass

from deeplotx.ner.n2g import Gender


@dataclass
class NamedEntity:
    text: str
    type: str
    base_probability: float


@dataclass
class NamedPerson(NamedEntity):
    gender: Gender
    gender_probability: float
