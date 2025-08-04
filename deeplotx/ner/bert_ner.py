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
        logger.debug(f'{BaseNER.__name__} initialized on device: {self.device}.')

    def extract_entities(self, s: str, with_gender: bool = True, prob_threshold: float = .0, *args, **kwargs) -> list[NamedEntity]:
        assert prob_threshold <= 1., f'prob_threshold ({prob_threshold}) cannot be larger than 1.'
        s = ' ' + s
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

    def __call__(self, s: str, with_gender: bool = True, prob_threshold: float = .0, *args, **kwargs):
        return self.extract_entities(s=s, with_gender=with_gender, prob_threshold=prob_threshold, *args, **kwargs)
