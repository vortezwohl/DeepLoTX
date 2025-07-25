import logging
import os
import math
from requests.exceptions import ConnectTimeout, SSLError

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from deeplotx import __ROOT__

CACHE_PATH = os.path.join(__ROOT__, '.cache')
DEFAULT_BERT = 'FacebookAI/xlm-roberta-base'
logger = logging.getLogger('deeplotx.embedding')


class Encoder(nn.Module):
    def __init__(self, model_name_or_path: str = DEFAULT_BERT, device: str | None = None):
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
        self.embed_dim = self.encoder.config.max_position_embeddings
        logger.debug(f'{Encoder.__name__} initialized on device: {self.device}.')

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        def _encoder(_input_tup: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            return self.encoder.forward(_input_tup[0], attention_mask=_input_tup[1]).last_hidden_state[:, 0, :]

        num_chunks = math.ceil(input_ids.shape[-1] / self.embed_dim)
        chunks = chunk_results = []
        for i in range(num_chunks):
            start_idx = i * self.embed_dim
            end_idx = min(start_idx + self.embed_dim, input_ids.shape[-1])
            chunks.append((input_ids[:, start_idx: end_idx], attention_mask[:, start_idx: end_idx]))
        ori_mode = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            chunk_results = [_encoder(x) for x in chunks]
        self.encoder.train(mode=ori_mode)
        return torch.cat(chunk_results, dim=-1)

    def encode(self, text: str) -> torch.Tensor:
        _input_ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long, device=self.device)
        _att_mask = torch.tensor([[1] * _input_ids.shape[-1]], dtype=torch.int, device=self.device)
        return self.forward(_input_ids, _att_mask).squeeze()
