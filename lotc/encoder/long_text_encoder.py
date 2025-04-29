import logging
from concurrent.futures import ThreadPoolExecutor
from typing_extensions import override

import torch

from lotc.encoder.bert_encoder import BertEncoder, DEFAULT_BERT
from lotc.util.hash import md5

logger = logging.getLogger('lotc.embedding')


class LongTextEncoder(BertEncoder):
    def __init__(self, max_length: int, chunk_size: int = 256,
                 overlapping: int = 0, model_name_or_path: str = DEFAULT_BERT):
        super().__init__(model_name_or_path=model_name_or_path)
        self._max_length = max_length
        self._chunk_size = chunk_size
        self._overlapping = overlapping
        self._cache = dict()

    def __chunk_embedding(self, input_tup: tuple[int, str]) -> tuple[int, torch.Tensor]:
        return input_tup[0], super().encode(input_tup[1])

    @override
    def encode(self, text: str) -> torch.Tensor:
        _text_to_show = text.replace("\n", str())
        logger.debug(f'Embedding \"{_text_to_show if len(_text_to_show) < 128 else _text_to_show[:128] + "..."}\".')
        # read cache
        _text_hash = md5(text)
        if _text_hash in self._cache.keys():
            return self._cache[_text_hash]
        _text = text.strip()
        if len(_text) < self._max_length:
            _text += '.' * (self._max_length - len(_text))
        num_chunks = max(int(self._max_length / self._chunk_size), 1)
        # split chunks
        chunks = []
        for i in range(num_chunks):
            _tmp_left = max(i * self._chunk_size - self._overlapping, 0)
            _tmp_right = (i + 1) * self._chunk_size + self._overlapping
            chunks.append((i, _text[_tmp_left: _tmp_right]))
        with ThreadPoolExecutor(max_workers=min(num_chunks + 1, 6)) as executor:
            embeddings = list(executor.map(self.__chunk_embedding, chunks))
        embeddings.sort(key=lambda x: x[0])
        fin_embedding = [x[1] for x in embeddings]
        fin_emb_tensor = torch.tensor([], dtype=torch.float32)
        for emb in fin_embedding:
            fin_emb_tensor = torch.cat((fin_emb_tensor.detach().clone(), emb.detach().clone()), dim=-1)
        # write cache
        self._cache[_text_hash] = fin_emb_tensor
        return fin_emb_tensor
