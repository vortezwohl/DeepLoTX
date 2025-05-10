import torch
from torch import nn
from transformers import LongformerTokenizer, LongformerModel

from deeplotx import __ROOT__

CACHE_PATH = f'{__ROOT__}\\.cache'
DEFAULT_LONGFORMER = 'allenai/longformer-base-4096'


class LongformerEncoder(nn.Module):
    def __init__(self, model_name_or_path: str = DEFAULT_LONGFORMER):
        super().__init__()
        self.tokenizer = LongformerTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                             cache_dir=CACHE_PATH, _from_auto=True)
        self.bert = LongformerModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                    cache_dir=CACHE_PATH, _from_auto=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
