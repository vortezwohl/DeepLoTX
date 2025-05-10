import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def z_score_normalize(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x)
    std_dev = np.std(x)
    return (x - mean) / std_dev


def euclidean_similarity(p: np.ndarray, q: np.ndarray) -> np.float32:
    distance = p - q
    distance = np.sum(np.multiply(distance, distance))
    return np.sqrt(distance)


def cosine_similarity(p: np.ndarray, q: np.ndarray) -> np.float32:
    a = np.matmul(np.transpose(p), q)
    b = np.sum(np.multiply(p, p))
    c = np.sum(np.multiply(q, q))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def chebyshev_similarity(p: np.ndarray, q: np.ndarray) -> np.float32:
    return np.max(np.abs(p - q))
