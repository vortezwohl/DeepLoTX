import logging
import math
from typing_extensions import override

import torch
from vortezwohl.concurrent import ThreadPool
from vortezwohl.cache import LRUCache

from deeplotx.encoder.encoder import Encoder, DEFAULT_BERT
from deeplotx.util.hash import sha512

logger = logging.getLogger('deeplotx.embedding')


class LongTextEncoder(Encoder):
    def __init__(self, max_length: int, chunk_size: int = 448,
                 overlapping: int = 32, model_name_or_path: str = DEFAULT_BERT,
                 cache_capacity: int = 64, max_workers: int = 8, device: str | None = None):
        super().__init__(model_name_or_path=model_name_or_path, device=device)
        self._max_length = max_length
        self._chunk_size = chunk_size
        self._overlapping = overlapping
        self._cache = LRUCache(capacity=cache_capacity)
        self._worker_group = ThreadPool(max_workers=max_workers)

    def __chunk_embedding(self, idx: int, x: torch.Tensor, mask: torch.Tensor) -> tuple[int, torch.Tensor]:
        return idx, super().forward(x, attention_mask=mask)

    @override
    def forward(self, text: str, flatten: bool = False, *args, **kwargs) -> torch.Tensor:
        return self.encode(text=text, flatten=flatten)

    @override
    def encode(self, text: str, flatten: bool = False) -> torch.Tensor:
        def postprocess(tensors: list[torch.Tensor], _flatten: bool) -> torch.Tensor:
            if not _flatten:
                return torch.stack(tensors, dim=0).squeeze()
            _fin_emb_tensor = torch.tensor([], dtype=tensors[0].dtype, device=self.device)
            for _emb in tensors:
                _fin_emb_tensor = torch.cat((_fin_emb_tensor.detach().clone(), _emb.detach().clone()), dim=-1)
            return _fin_emb_tensor.squeeze()

        _text_to_show = text.replace("\n", str())
        logger.debug(f'Embedding \"{_text_to_show if len(_text_to_show) < 128 else _text_to_show[:128] + "..."}\".')
        # read cache
        _text_hash = sha512(text)
        if _text_hash in self._cache:
            return postprocess(self._cache[_text_hash], flatten)
        _text_to_input_ids = self.tokenizer.encode(text.strip())[:self._max_length]
        _text_to_input_ids_att_mask = []
        # padding
        pad_token = self.tokenizer.pad_token_type_id
        if len(_text_to_input_ids) < self._max_length:
            _text_to_input_ids.extend([pad_token] * (self._max_length - len(_text_to_input_ids)))
        pads = _text_to_input_ids.count(pad_token)
        non_pads = self._max_length - pads
        _text_to_input_ids_att_mask.extend([1] * non_pads)
        _text_to_input_ids_att_mask.extend([0] * pads)
        num_chunks = math.ceil(self._max_length / self._chunk_size)
        # split chunks
        chunks = []
        for i in range(num_chunks):
            _tmp_left = max(i * self._chunk_size - self._overlapping, 0)
            _tmp_right = (i + 1) * self._chunk_size + self._overlapping
            chunks.append((i, torch.tensor([_text_to_input_ids[_tmp_left: _tmp_right]], dtype=torch.int, device=self.device),
                           torch.tensor([_text_to_input_ids_att_mask[_tmp_left: _tmp_right]], dtype=torch.int, device=self.device)))
        embeddings = list(self._worker_group.map(self.__chunk_embedding, chunks))
        embeddings = sorted([x.returns for x in embeddings], key=lambda x: x[0], reverse=False)
        fin_embedding = [x[1] for x in embeddings]
        # write cache
        self._cache[_text_hash] = fin_embedding
        return postprocess(fin_embedding, flatten)
