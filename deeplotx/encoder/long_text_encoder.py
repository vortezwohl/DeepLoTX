import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing_extensions import override

import torch

from deeplotx.encoder.bert_encoder import BertEncoder, DEFAULT_BERT
from deeplotx.util.hash import md5

logger = logging.getLogger('deeplotx.embedding')


class LongTextEncoder(BertEncoder):
    def __init__(self, max_length: int, chunk_size: int = 256,
                 overlapping: int = 0, model_name_or_path: str = DEFAULT_BERT):
        super().__init__(model_name_or_path=model_name_or_path)
        self._max_length = max_length
        self._chunk_size = chunk_size
        self._overlapping = overlapping
        self._cache = dict()

    def __chunk_embedding(self, input_tup: tuple[int, torch.Tensor]) -> tuple[int, torch.Tensor]:
        return input_tup[0], super().forward(input_tup[1], attention_mask=input_tup[2])

    @override
    def encode(self, text: str, use_cache: bool = True) -> torch.Tensor:
        _text_to_show = text.replace("\n", str())
        logger.debug(f'Embedding \"{_text_to_show if len(_text_to_show) < 128 else _text_to_show[:128] + "..."}\".')
        # read cache
        _text_hash = md5(text)
        if _text_hash in self._cache.keys():
            return self._cache[_text_hash]
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
            chunks.append((i, torch.tensor([_text_to_input_ids[_tmp_left: _tmp_right]], dtype=torch.long),
                           torch.tensor([_text_to_input_ids_att_mask[_tmp_left: _tmp_right]], dtype=torch.int)))
        with ThreadPoolExecutor(max_workers=min(num_chunks + 1, 3)) as executor:
            embeddings = list(executor.map(self.__chunk_embedding, chunks))
        embeddings.sort(key=lambda x: x[0])
        fin_embedding = [x[1] for x in embeddings]
        fin_emb_tensor = torch.tensor([], dtype=torch.float32)
        for emb in fin_embedding:
            fin_emb_tensor = torch.cat((fin_emb_tensor.detach().clone(), emb.detach().clone()), dim=-1)
        fin_emb_tensor = fin_emb_tensor.squeeze()
        # write cache
        if use_cache:
            self._cache[_text_hash] = fin_emb_tensor
        return fin_emb_tensor
