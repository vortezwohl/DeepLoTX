import numpy as np
import torch

from deeplotx.similarity import ndarray_adapter


def l2_normalize(x: torch.Tensor | np.ndarray) -> np.ndarray:
    x = ndarray_adapter(x)
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def z_score_normalize(x: torch.Tensor | np.ndarray) -> np.ndarray:
    x = ndarray_adapter(x)
    mean = np.mean(x)
    std_dev = np.std(x)
    return (x - mean) / std_dev


def euclidean_similarity(p: torch.Tensor | np.ndarray, q: torch.Tensor | np.ndarray) -> np.float32:
    p, q = ndarray_adapter(p, q)
    distance = p - q
    distance = np.sum(np.multiply(distance, distance))
    return np.sqrt(distance)


def cosine_similarity(p: torch.Tensor | np.ndarray, q: torch.Tensor | np.ndarray) -> np.float32:
    p, q = ndarray_adapter(p, q)
    a = np.matmul(np.transpose(p), q)
    b = np.sum(np.multiply(p, p))
    c = np.sum(np.multiply(q, q))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def chebyshev_similarity(p: torch.Tensor | np.ndarray, q: torch.Tensor | np.ndarray) -> np.float32:
    p, q = ndarray_adapter(p, q)
    return np.max(np.abs(p - q))
