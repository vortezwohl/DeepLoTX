import numpy as np
import torch

bias = 1e-12


def ndarray_adapter(*args) -> tuple | np.ndarray:
    args = list(args)
    for i, arg in enumerate(args):
        match arg.__class__:
            case torch.Tensor:
                args[i] = arg.detach().cpu().numpy()
            case List:
                args[i] = np.asarray(arg)
    if len(args) > 1:
        return tuple(args)
    return args[0]


from .set import jaccard_similarity, ochiai_similarity, dice_coefficient, overlap_coefficient
from .vector import euclidean_similarity, cosine_similarity, chebyshev_similarity, l2_normalize, z_score_normalize
from .distribution import cross_entropy, kl_divergence, js_divergence, hellinger_distance
