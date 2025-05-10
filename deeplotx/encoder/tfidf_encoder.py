import math

import torch

from deeplotx.encoder.bert_encoder import DEFAULT_BERT, BertEncoder


class TfIdfEncoder(BertEncoder):
    def __init__(self, tokenizer_name_or_path: str = DEFAULT_BERT):
        super().__init__(model_name_or_path=tokenizer_name_or_path)

    def tokenize(self, text: str) -> tuple[list[str], list[int]]:
        tokens = self.tokenizer.tokenize(text)
        return tokens, self.tokenizer.convert_tokens_to_ids(tokens)

    def tf(self, term: str | int, document: str | list[int], mode: str = 'log') -> float | int:
        term_count = document.count(term)
        match mode:
            case 'simple':
                return term_count
            case 'log':
                return math.log(term_count + 1)
            case 'bool':
                return 1 if term_count in document else 0
            case _:
                # proportion based
                all_terms = len(self.tokenize(document))
                return term_count / all_terms

    def idf(self, term: str | int, documents: list[str] | list[list[int]], mode: str = 'smooth') -> float:
        all_documents = len(documents)
        documents_contains_term = len([doc for doc in documents if term in doc])
        match mode:
            case 'simple':
                return math.log(all_documents / documents_contains_term)
            case 'smooth':
                return math.log((all_documents + 1) / (documents_contains_term + 1))
            case _:
                # probability based
                return math.log((all_documents - documents_contains_term + 0.5) / (documents_contains_term + 0.5))

    def encode(self, text: str) -> torch.Tensor:


