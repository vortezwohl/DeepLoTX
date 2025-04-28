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
        self.bert.eval()
        with torch.no_grad():
            res = self.bert.forward(input_ids)
        return res

    # noinspection PyUnresolvedReferences
    def encode(self, text: str) -> torch.Tensor:
        _input_ids = self.tokenizer.encode(text, return_tensors='pt')
        return self.forward(_input_ids).last_hidden_state[:, 0, :].squeeze()
