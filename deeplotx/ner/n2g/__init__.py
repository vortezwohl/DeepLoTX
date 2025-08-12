import os

import requests
import gdown
import torch
from name4py import Gender

from deeplotx import __ROOT__
from deeplotx.encoder.encoder import Encoder
from deeplotx.nn.logistic_regression import LogisticRegression
from deeplotx.nn.base_neural_network import BaseNeuralNetwork


__CACHE_DIR__ = os.path.join(__ROOT__, '.cache', '.n2g')
ENCODER = Encoder(model_name_or_path='FacebookAI/xlm-roberta-base')
DEFAULT_MODEL = 'name2gender-small'
_MIN_FILE_SIZE = 1024 * 5


def download_model(model_name: str):
    quiet = bool(os.getenv('N2G_QUIET_DOWNLOAD', False))
    os.makedirs(__CACHE_DIR__, exist_ok=True)
    _proxies = {
        'http': os.getenv('HTTP_PROXY', os.getenv('http_proxy')),
        'https': os.getenv('HTTPS_PROXY', os.getenv('https_proxy'))
    }
    model_name = f'{model_name}.dlx'
    model_path = os.path.join(__CACHE_DIR__, model_name)
    base_url = 'https://github.com/vortezwohl/Name2Gender'
    if not os.path.exists(model_path):
        url = f'{base_url}/releases/download/RESOURCE/{model_name}'
        if requests.get(url=base_url, proxies=_proxies).status_code == 200:
            try:
                gdown.download(
                    url=url,
                    output=model_path,
                    quiet=quiet,
                    proxy=_proxies.get('https'),
                    speed=8192 * 1024,
                    resume=True
                )
                if os.path.getsize(model_path) < _MIN_FILE_SIZE:
                    raise FileNotFoundError(f"Model \"{model_name}\" doesn't exists.")
            except Exception as e:
                os.remove(model_path)
                raise e
        else:
            raise ConnectionError(f'Failed to download model {model_name}.')


def load_model(model_name: str = 'name2gender-small', dtype: torch.dtype | None = torch.float16) -> BaseNeuralNetwork:
    n2g_model = None
    if 'base' in model_name.lower():
        download_model(model_name)
        n2g_model = LogisticRegression(input_dim=768, output_dim=1,
                                       num_heads=12, num_layers=4,
                                       head_layers=1, expansion_factor=2,
                                       model_name=model_name, dtype=dtype)
    elif 'small' in model_name.lower():
        download_model(model_name)
        n2g_model = LogisticRegression(input_dim=768, output_dim=1,
                                       num_heads=6, num_layers=2,
                                       head_layers=1, expansion_factor=1.5,
                                       model_name=model_name, dtype=dtype)
    else:
        raise FileNotFoundError(f"Model \"{model_name}\" doesn't exists.")
    return n2g_model.load(model_dir=__CACHE_DIR__)


class Name2Gender:
    def __init__(self, model: BaseNeuralNetwork | None = None):
        super().__init__()
        if model is None:
            model = load_model(DEFAULT_MODEL)
        self._model = model

    def __call__(self, name: str, return_probability: bool = False, threshold: float = .5) -> tuple[Gender, float] | Gender:
        assert len(name) > 0, f'name ({name}) cannot be empty.'
        name = f'{name[0].upper()}{name[1:]}'
        emb = ENCODER.encode(name)
        prob = self._model.predict(emb).item()
        gender = Gender.Male if prob >= threshold else Gender.Female
        if return_probability:
            return gender, prob if gender == Gender.Male else (1. - prob)
        return gender
