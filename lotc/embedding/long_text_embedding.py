import logging
from concurrent.futures import ThreadPoolExecutor

import torch
from sentence_transformers import SentenceTransformer

from lotc import __ROOT__
from lotc.util.hash import md5

CACHE = dict()
logger = logging.getLogger('lotc.embedding')


def long_text_embedding(text: str, max_length: int,
                        chunk_size: int = 256,
                        bert_model_name_or_path: str = 'moka-ai/m3e-small') -> tuple[int, torch.Tensor]:
    def chunk_embedding(input_tup: tuple[int, str]) -> tuple[int, torch.Tensor]:
        return input_tup[0], bert_model.encode(input_tup[1])

    _text_to_show = text.replace("\n", str())
    logger.debug(f'Embedding \"{_text_to_show if len(_text_to_show) < 128 else _text_to_show[:128] + "..."}\".')
    # read cache
    _text_hash = md5(text)
    if _text_hash in CACHE.keys():
        _text_cache_dict = CACHE[_text_hash]
        if (_text_cache_dict.get('max_length') == max_length
                and _text_cache_dict.get('chunk_size') == chunk_size
                and _text_cache_dict.get('bert_model_name_or_path') == bert_model_name_or_path):
            if 'result' in _text_cache_dict.keys():
                return _text_cache_dict.get('result')
    bert_model = SentenceTransformer(
        model_name_or_path=bert_model_name_or_path,
        cache_folder=f'{__ROOT__}\\.cache'
    )
    _text = text.strip()
    if len(_text) < max_length:
        _text += '.' * (max_length - len(_text))
    num_chunks = max(int(max_length / chunk_size), 1)
    chunks = []
    for i in range(num_chunks):
        chunks.append((i, _text[i * chunk_size: (i + 1) * chunk_size]))
    with ThreadPoolExecutor(max_workers=min(num_chunks + 1, 8)) as executor:
        embeddings = list(executor.map(chunk_embedding, chunks))
    embeddings.sort(key=lambda x: x[0])
    fin_embedding = [x[1] for x in embeddings]
    fin_emb_tensor = torch.tensor([], dtype=torch.float32)
    for emb in fin_embedding:
        fin_emb_tensor = torch.cat((fin_emb_tensor, torch.tensor(emb, dtype=torch.float32)), dim=0)
    result = fin_emb_tensor.shape[-1], fin_emb_tensor
    # write cache
    CACHE[_text_hash] = {
        'max_length': max_length,
        'chunk_size': chunk_size,
        'bert_model_name_or_path': bert_model_name_or_path,
        'result': result
    }
    return result
