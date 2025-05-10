import math
import torch
from torch import nn
from transformers import BertTokenizer, BertModel

from deeplotx import __ROOT__

CACHE_PATH = f'{__ROOT__}\\.cache'
DEFAULT_BERT = 'bert-base-uncased'


class BertEncoder(nn.Module):
    def __init__(self, model_name_or_path: str = DEFAULT_BERT):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                       cache_dir=CACHE_PATH, _from_auto=True)
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                              cache_dir=CACHE_PATH, _from_auto=True)
        self.embed_dim = self.bert.config.max_position_embeddings

    def forward(self, input_ids, attention_mask: torch.Tensor) -> torch.Tensor:
        def _encoder(_input_tup: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            return self.bert.forward(_input_tup[0], attention_mask=_input_tup[1]).last_hidden_state[:, 0, :]

        num_chunks = math.ceil(input_ids.shape[-1] / self.embed_dim)
        chunks = chunk_results = []
        for i in range(num_chunks):
            start_idx = i * self.embed_dim
            end_idx = min(start_idx + self.embed_dim, input_ids.shape[-1])
            chunks.append((input_ids[:, start_idx: end_idx], attention_mask[:, start_idx: end_idx]))
        ori_mode = self.bert.training
        self.bert.eval()
        with torch.no_grad():
            chunk_results = [_encoder(x) for x in chunks]
        self.bert.train(mode=ori_mode)
        return torch.cat(chunk_results, dim=-1)

    def encode(self, text: str) -> torch.Tensor:
        _input_ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)
        _att_mask = torch.tensor([[1] * _input_ids.shape[-1]], dtype=torch.int)
        return self.forward(_input_ids, _att_mask).squeeze()
