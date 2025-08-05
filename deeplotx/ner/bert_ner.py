import logging
import os
from requests.exceptions import ConnectTimeout, SSLError

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from deeplotx import __ROOT__
from deeplotx.ner.n2g import Name2Gender
from deeplotx.ner.base_ner import BaseNER
from deeplotx.ner.named_entity import NamedEntity, NamedPerson

CACHE_PATH = os.path.join(__ROOT__, '.cache')
DEFAULT_BERT_NER = 'Davlan/xlm-roberta-base-ner-hrl'
N2G_MODEL: list[Name2Gender] = []
logger = logging.getLogger('deeplotx.ner')


class BertNER(BaseNER):
    def __init__(self, model_name_or_path: str = DEFAULT_BERT_NER, device: str | None = None):
        super().__init__()
        self.device = torch.device(device) if device is not None \
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                           cache_dir=CACHE_PATH, _from_auto=True,
                                                           trust_remote_code=True)
            self.encoder = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                                           cache_dir=CACHE_PATH, _from_auto=True,
                                                                           trust_remote_code=True).to(self.device)
        except ConnectTimeout:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                           cache_dir=CACHE_PATH, _from_auto=True,
                                                           trust_remote_code=True, local_files_only=True)
            self.encoder = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                                           cache_dir=CACHE_PATH, _from_auto=True,
                                                                           trust_remote_code=True, local_files_only=True).to(self.device)
        except SSLError:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                           cache_dir=CACHE_PATH, _from_auto=True,
                                                           trust_remote_code=True, local_files_only=True)
            self.encoder = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                                           cache_dir=CACHE_PATH, _from_auto=True,
                                                                           trust_remote_code=True, local_files_only=True).to(self.device)
        self.embed_dim = self.encoder.config.max_position_embeddings
        self._ner_pipeline = pipeline(task='ner', model=self.encoder, tokenizer=self.tokenizer, trust_remote_code=True)
        logger.debug(f'{BertNER.__name__} initialized on device: {self.device}.')

    def _fast_extract(self, s: str, with_gender: bool = True, prob_threshold: float = .0) -> list[NamedEntity]:
        assert prob_threshold <= 1., f'prob_threshold ({prob_threshold}) cannot be larger than 1.'
        s = f' {s} '
        raw_entities = self._ner_pipeline(s)
        entities = []
        for ent in raw_entities:
            entities.append([s[ent['start']: ent['end']], ent['entity'], ent['score'].item()])
        while True:
            for i, ent in enumerate(entities):
                if len(ent[0].strip()) < 1:
                    del entities[i]
                if ent[1].upper().startswith('I') and entities[i - 1][1].upper().startswith('B'):
                    entities[i - 1][0] += ent[0]
                    entities[i - 1][2] *= ent[2]
                    del entities[i]
            _continue = False
            for ent in entities:
                if ent[1].upper().startswith('I'):
                    _continue = True
            if not _continue:
                break
        for ent in entities:
            ent[0] = ent[0].strip()
            if ent[1].upper().startswith('B'):
                ent[1] = ent[1].upper()[1:].strip('-')
        entities = [NamedEntity(*_) for _ in entities if _[2] >= prob_threshold]
        if not with_gender:
            return entities
        if len(N2G_MODEL) < 1:
            N2G_MODEL.append(Name2Gender())
        n2g_model = N2G_MODEL[0]
        for i, ent in enumerate(entities):
            if ent.type.upper() == 'PER':
                gender, gender_prob = n2g_model(ent.text, return_probability=True)
                entities[i] = NamedPerson(text=ent.text,
                                          type=ent.type,
                                          base_probability=ent.base_probability,
                                          gender=gender,
                                          gender_probability=gender_prob)
        return entities

    def _slow_extract(self, s: str, with_gender: bool = True, prob_threshold: float = .0, deduplicate: bool = True) -> list[NamedEntity]:
        _entities = self._fast_extract(s, with_gender=with_gender, prob_threshold=prob_threshold) if len(s) < 512 else []
        if len(s) >= 512:
            window_size: int = 512
            offset = window_size // 6
            for _offset in [- offset, offset]:
                _window_size = window_size + _offset
                for i in range(0, len(s) + _window_size, _window_size):
                    _entities.extend(self._fast_extract(s[i: i + _window_size], with_gender=with_gender, prob_threshold=prob_threshold))
        _tmp_entities = sorted(_entities, key=lambda x: len(x.text), reverse=True)
        for _ent_i in _tmp_entities:
            for _ent_j in _entities:
                if (_ent_j.text in _ent_i.text
                        and len(_ent_j.text) != len(_ent_i.text)
                        and _ent_j in _tmp_entities):
                    _tmp_entities.remove(_ent_j)
        while True:
            for _ent in _tmp_entities:
                if _ent.text not in s or len(_ent.text) < 2:
                    _tmp_entities.remove(_ent)
            _continue = False
            for _ent in _tmp_entities:
                if _ent.text not in s or len(_ent.text) < 2:
                    _continue = True
                    break
            if not _continue:
                break
        if not deduplicate:
            return sorted(_tmp_entities, key=lambda _: _.text[0], reverse=False)
        _fin_entities = dict()
        texts = set([text.text for text in _tmp_entities])
        for text in texts:
            for _ent in _tmp_entities:
                if _ent.text == text:
                    if _ent.text not in _fin_entities.keys():
                        _fin_entities[_ent.text] = _ent
                    else:
                        if _ent.base_probability > _fin_entities[_ent.text].base_probability:
                            _fin_entities[_ent.text] = _ent
        return sorted([v for k, v in _fin_entities.items()], key=lambda _: _.text[0], reverse=False)

    def __call__(self, s: str, with_gender: bool = True, prob_threshold: float = .0, fast_mode: bool = False, *args, **kwargs):
        if fast_mode:
            return self._fast_extract(s=s, with_gender=with_gender, prob_threshold=prob_threshold)
        else:
            return self._slow_extract(s=s, with_gender=with_gender, prob_threshold=prob_threshold,
                                      deduplicate=kwargs.get('deduplicate', True))
