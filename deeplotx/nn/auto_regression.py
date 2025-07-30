import torch

from deeplotx.nn import RecursiveSequential


class AutoRegression(RecursiveSequential):
    def __init__(self, feature_dim: int, bias: bool = True,
                 recursive_layers: int = 1, recursive_hidden_dim: int | None = None,
                 ffn_layers: int = 1, ffn_expansion_factor: int | float = 2, dropout_rate: float = 0.05,
                 model_name: str | None = None, device: str | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(input_dim=feature_dim, output_dim=feature_dim, bias=bias,
                         recursive_layers=recursive_layers, recursive_hidden_dim=recursive_hidden_dim,
                         ffn_layers=ffn_layers, ffn_expansion_factor=ffn_expansion_factor,
                         dropout_rate=dropout_rate, model_name=model_name, device=device, dtype=dtype, **kwargs)
