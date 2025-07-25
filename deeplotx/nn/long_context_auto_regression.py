import torch

from deeplotx.nn import LongContextRecursiveSequential


class LongContextAutoRegression(LongContextRecursiveSequential):
    def __init__(self, feature_dim: int, hidden_dim: int | None = None,
                 recursive_layers: int = 2, ffn_layers: int = 1, ffn_expansion_factor: int | float = 2,
                 ffn_bias: bool = True, ffn_dropout_rate: float = 0.05,  model_name: str | None = None,
                 device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__(input_dim=feature_dim, output_dim=feature_dim,
                         hidden_dim=hidden_dim, recursive_layers=recursive_layers,
                         ffn_layers=ffn_layers, ffn_expansion_factor=ffn_expansion_factor,
                         ffn_bias=ffn_bias, ffn_dropout_rate=ffn_dropout_rate,
                         model_name=model_name, device=device, dtype=dtype)
