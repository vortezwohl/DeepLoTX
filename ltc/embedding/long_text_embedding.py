from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from sentence_transformers import SentenceTransformer


def long_text_embedding(text: str, max_length: int,
                        chunk_size: int = 512, bert_model: SentenceTransformer = None) -> tuple[int, torch.Tensor]:
    def chunk_embedding(input_tup: tuple[int, str]) -> tuple[int, torch.Tensor]:
        return input_tup[0], bert_model.encode(input_tup[1])

    if bert_model is None:
        bert_model = SentenceTransformer('moka-ai/m3e-small')
    _text = text.strip()
    if len(_text) < max_length:
        _text += '.' * (max_length - len(_text))
    num_chunks = max(int(max_length / chunk_size), 1)
    chunks = []
    for i in range(num_chunks):
        chunks.append((i, _text[i * chunk_size: (i + 1) * chunk_size]))
    with ThreadPoolExecutor(max_workers=min(num_chunks + 1, 6)) as executor:
        embeddings = list(executor.map(chunk_embedding, chunks))
    embeddings.sort(key=lambda x: x[0])
    fin_embedding = [x[1] for x in embeddings]
    fin_emb_tensor = torch.tensor([], dtype=torch.float32)
    for emb in fin_embedding:
        fin_emb_tensor = torch.cat((fin_emb_tensor, torch.tensor(emb, dtype=torch.float32)), dim=0)
    return fin_emb_tensor.shape[-1], fin_emb_tensor


if __name__ == '__main__':
    res = long_text_embedding('niha0', max_length=1050)
    print(res)
