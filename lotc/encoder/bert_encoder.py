from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn
from transformers import BertTokenizer, BertModel

from lotc import __ROOT__

CACHE_PATH = f'{__ROOT__}\\.cache'
DEFAULT_BERT = 'bert-base-uncased'


class BertEncoder(nn.Module):
    def __init__(self, model_name_or_path: str = DEFAULT_BERT):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                       cache_dir=CACHE_PATH)
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                              cache_dir=CACHE_PATH)

    def forward(self, input_ids) -> torch.Tensor:
        ori_mode = self.bert.training
        self.bert.eval()
        num_chunks = (input_ids.shape[-1] + 511) // 512
        chunks = chunk_results = []
        for i in range(num_chunks):
            start_idx = i * 512
            end_idx = min(start_idx + 512, input_ids.shape[1])
            chunks.append(input_ids[:, start_idx: end_idx])
        with torch.no_grad():
            with ThreadPoolExecutor(max_workers=4) as e:
                chunk_results = [x.last_hidden_state[:, 0, :]
                                 for x in list(e.map(self.bert.forward, chunks))]
        self.bert.train(mode=ori_mode)
        return torch.mean(torch.stack(chunk_results), dim=0)

    # noinspection PyUnresolvedReferences
    def encode(self, text: str) -> torch.Tensor:
        _input_ids = self.tokenizer.encode(text, return_tensors='pt')
        return self.forward(_input_ids)
