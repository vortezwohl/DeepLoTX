from deeplotx.nn import RecursiveSequential


class AutoRegression(RecursiveSequential):
    def __init__(self, feature_dim: int, hidden_dim: int | None = None,
                 recursive_layers: int = 2, model_name: str | None = None):
        super().__init__(input_dim=feature_dim, output_dim=feature_dim,
                         hidden_dim=hidden_dim, recursive_layers=recursive_layers,
                         model_name=model_name)
