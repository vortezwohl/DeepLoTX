import numpy as np
import torch

from deeplotx.similarity import bias, ndarray_adapter


def cross_entropy(p: torch.Tensor | np.ndarray, q: torch.Tensor | np.ndarray) -> np.float32:
    p, q = ndarray_adapter(p, q)
    q = np.clip(q, bias, 1 - bias)
    return -1 * (np.sum(p * np.log(q)) / p.shape[0])


def kl_divergence(p: torch.Tensor | np.ndarray, q: torch.Tensor | np.ndarray) -> np.float32:
    p, q = ndarray_adapter(p, q)
    q = np.where(q == 0, bias, q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))


def js_divergence(p: torch.Tensor | np.ndarray, q: torch.Tensor | np.ndarray) -> np.float32:
    p, q = ndarray_adapter(p, q)
    m = (p + q) / 2
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2


def hellinger_distance(p: torch.Tensor | np.ndarray, q: torch.Tensor | np.ndarray) -> np.float32:
    p, q = ndarray_adapter(p, q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    squared_diff = (np.sqrt(p) - np.sqrt(q)) ** 2
    return np.sqrt(np.sum(squared_diff)) / np.sqrt(2)
