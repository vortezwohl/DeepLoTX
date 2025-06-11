import logging
import os

import torch
from torch import nn
from transformers import LongformerTokenizer, LongformerModel

from deeplotx import __ROOT__

CACHE_PATH = os.path.join(__ROOT__, '.cache')
DEFAULT_LONGFORMER = 'allenai/longformer-base-4096'
logger = logging.getLogger('deeplotx.embedding')


class LongformerEncoder(nn.Module):
    def __init__(self, model_name_or_path: str = DEFAULT_LONGFORMER, device: str | None = None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = LongformerTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                             cache_dir=CACHE_PATH, _from_auto=True)
        self.bert = LongformerModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                    cache_dir=CACHE_PATH, _from_auto=True).to(self.device)
        logger.debug(f'{LongformerEncoder.__name__} initialized on device: {self.device}.')

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        ori_mode = self.bert.training
        self.bert.eval()
        with torch.no_grad():
            res = self.bert.forward(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        self.bert.train(mode=ori_mode)
        return res

    def encode(self, text: str) -> torch.Tensor:
        _input_ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)
        _att_mask = torch.tensor([[1] * _input_ids.shape[-1]], dtype=torch.int)
        return self.forward(_input_ids, _att_mask).squeeze()
