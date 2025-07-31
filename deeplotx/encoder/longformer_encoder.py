import logging
import os

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from requests.exceptions import ConnectTimeout, SSLError

from deeplotx import __ROOT__

CACHE_PATH = os.path.join(__ROOT__, '.cache')
DEFAULT_LONGFORMER = 'severinsimmler/xlm-roberta-longformer-base-16384'
logger = logging.getLogger('deeplotx.embedding')


class LongformerEncoder(nn.Module):
    def __init__(self, model_name_or_path: str = DEFAULT_LONGFORMER, device: str | None = None):
        super().__init__()
        self.device = torch.device(device) if device is not None \
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                           cache_dir=CACHE_PATH, _from_auto=True,
                                                           trust_remote_code=True)
            self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                     cache_dir=CACHE_PATH, _from_auto=True,
                                                     trust_remote_code=True).to(self.device)
        except ConnectTimeout:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                           cache_dir=CACHE_PATH, _from_auto=True,
                                                           trust_remote_code=True, local_files_only=True)
            self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                     cache_dir=CACHE_PATH, _from_auto=True,
                                                     trust_remote_code=True, local_files_only=True).to(self.device)
        except SSLError:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                           cache_dir=CACHE_PATH, _from_auto=True,
                                                           trust_remote_code=True, local_files_only=True)
            self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                     cache_dir=CACHE_PATH, _from_auto=True,
                                                     trust_remote_code=True, local_files_only=True).to(self.device)
        logger.debug(f'{LongformerEncoder.__name__} initialized on device: {self.device}.')

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, cls_only: bool = True) -> torch.Tensor:
        ori_mode = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            emb_seq = self.encoder.forward(input_ids, attention_mask=attention_mask).last_hidden_state
            res = emb_seq[:, 0, :] if cls_only else emb_seq
        self.encoder.train(mode=ori_mode)
        return res

    def encode(self, text: str, cls_only: bool = True) -> torch.Tensor:
        _input_ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long, device=self.device)
        _att_mask = torch.tensor([[1] * _input_ids.shape[-1]], dtype=torch.int, device=self.device)
        return self.forward(input_ids=_input_ids, attention_mask=_att_mask, cls_only=cls_only).squeeze()
