import numpy as np

from deeplotx.similarity import bias


def cross_entropy(p: np.ndarray, q: np.ndarray) -> np.float32:
    q = np.clip(q, bias, 1 - bias)
    return -1 * (np.sum(p * np.log(q)) / p.shape[0])


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.float32:
    q = np.where(q == 0, bias, q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.float32:
    m = (p + q) / 2
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> np.float32:
    p = p / np.sum(p)
    q = q / np.sum(q)
    squared_diff = (np.sqrt(p) - np.sqrt(q)) ** 2
    return np.sqrt(np.sum(squared_diff)) / np.sqrt(2)
