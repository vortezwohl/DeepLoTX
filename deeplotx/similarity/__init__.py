import numpy as np
import torch

bias = 1e-12


def ndarray_adapter(p: any, q: any) -> tuple[np.ndarray, np.ndarray]:
    match p.__class__:
        case torch.Tensor:
            p = p.detach().cpu().numpy()
        case List:
            p = np.asarray(p)
    match q.__class__:
        case torch.Tensor:
            q = q.detach().cpu().numpy()
        case List:
            q = np.asarray(q)
    return p, q
