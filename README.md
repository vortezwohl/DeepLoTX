[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/vortezwohl/DeepLoTX)

# *Deep Long Text Learning*

*An out-of-the-box long-text NLP framework.*

> Author: [vortezwohl](https://github.com/vortezwohl)

## Citation

If you are incorporating the `DeepLoTX` framework into your research, please remember to properly cite it to acknowledge its contribution to your work.

Если вы интегрируете фреймворк `DeepLoTX` в своё исследование, пожалуйста, не забудьте правильно сослаться на него, указывая его вклад в вашу работу.

もしあなたが研究に `DeepLoTX` フレームワークを組み入れているなら、その貢献を認めるために適切に引用することを忘れないでください.

如果您正在將 `DeepLoTX` 框架整合到您的研究中，請務必正確引用它，以聲明它對您工作的貢獻.

```bibtex
@software{Wu_DeepLoTX_2025,
author = {Wu, Zihao},
license = {GPL-3.0},
month = aug,
title = {{DeepLoTX}},
url = {https://github.com/vortezwohl/DeepLoTX},
version = {0.9.5},
year = {2025}
}
```

## Installation

- **With pip**

    ```
    pip install -U deeplotx
    ```

- **With uv (recommended)**

    ```
    uv add -U deeplotx
    ```
  
- **Get the latest features from GitHub**

    ```
    pip install -U git+https://github.com/vortezwohl/DeepLoTX.git
    ```

## Quick start

- ### Named entity recognition

    > *Multilingual is supported.*

    > *Gender recognition is supported.*

    Import dependencies

    ```python
    from deeplotx import BertNER

    ner = BertNER()
    ```

    ```python
    ner('你好, 我的名字是吴子豪, 来自福建福州.')
    ```

    stdout:

    ```
    [NamedPerson(text='吴子豪', type='PER', base_probability=0.9995428418719051, gender=<Gender.Male: 'male'>, gender_probability=0.9970703125),
    NamedEntity(text='福建', type='LOC', base_probability=0.9986373782157898),
    NamedEntity(text='福州', type='LOC', base_probability=0.9993632435798645)]
    ```

    ```python
    ner("Hi, i'm Vortez Wohl, author of DeeploTX.")
    ```

    stdout:

    ```
    [NamedPerson(text='Vortez Wohl', type='PER', base_probability=0.9991965342072855, gender=<Gender.Male: 'male'>, gender_probability=0.87255859375)]
    ```

- ### Gender recognition

    > *Multilingual is supported.*

    > *Integrated from [Name2Gender](https://github.com/vortezwohl/Name2Gender)*

    Import dependencies

    ```python
    from deeplotx import Name2Gender

    n2g = Name2Gender()
    ```

    Recognize gender of "Elon Musk":

    ```python
    n2g('Elon Musk')
    ```

    stdout:

    ```
    <Gender.Male: 'male'>
    ```

    Recognize gender of "Anne Hathaway":

    ```python
    n2g('Anne Hathaway')
    ```

    stdout:

    ```
    <Gender.Female: 'female'>
    ```

    Recognize gender of "吴彦祖":

    ```python
    n2g('吴彦祖', return_probability=True)
    ```

    stdout:

    ```
    (<Gender.Male: 'male'>, 1.0)
    ```

- ### Long text embedding

    - **BERT based long text embedding**

        ```python
        from deeplotx import LongTextEncoder

        encoder = LongTextEncoder(
            chunk_size=448,
            overlapping=32
        )
        encoder.encode('我是吴子豪, 这是一个测试文本.', flatten=False)
        ```

        stdout:
        ```
        tensor([ 2.2316e-01,  2.0300e-01,  ...,  1.5578e-01, -6.6735e-02])
        ```

    - **Longformer based long text embedding**

        ```python
        from deeplotx import LongformerEncoder

        encoder = LongformerEncoder()
        encoder.encode('Thank you for using DeepLoTX.')
        ```

        stdout:
        ```
        tensor([-2.7490e-02,  6.6503e-02, ..., -6.5937e-02,  6.7802e-03])
        ```

- ### Similarities calculation

    - **Vector based**

        ```python
        import deeplotx.similarity as sim

        vector_0, vector_1 = [1, 2, 3, 4], [4, 3, 2, 1]
        distance_0 = sim.euclidean_similarity(vector_0, vector_1)
        print(distance_0)
        distance_1 = sim.cosine_similarity(vector_0, vector_1)
        print(distance_1)
        distance_2 = sim.chebyshev_similarity(vector_0, vector_1)
        print(distance_2)
        ```

        stdout:
        ```
        4.47213595499958
        0.33333333333333337
        3
        ```

    - **Set based**

        ```python
        import deeplotx.similarity as sim

        set_0, set_1 = {1, 2, 3, 4}, {4, 5, 6, 7}
        distance_0 = sim.jaccard_similarity(set_0, set_1)
        print(distance_0)
        distance_1 = sim.ochiai_similarity(set_0, set_1)
        print(distance_1)
        distance_2 = sim.dice_coefficient(set_0, set_1)
        print(distance_2)
        distance_3 = sim.overlap_coefficient(set_0, set_1)
        print(distance_3)
        ```

        stdout:
        ```
        0.1428571428572653
        0.2500000000001875
        0.25000000000009376
        0.2500000000001875
        ```

    - **Distribution based**

        ```python
        import deeplotx.similarity as sim

        dist_0, dist_1 = [0.3, 0.2, 0.1, 0.4], [0.2, 0.1, 0.3, 0.4]
        distance_0 = sim.cross_entropy(dist_0, dist_1)
        print(distance_0)
        distance_1 = sim.kl_divergence(dist_0, dist_1)
        print(distance_1)
        distance_2 = sim.js_divergence(dist_0, dist_1)
        print(distance_2)
        distance_3 = sim.hellinger_distance(dist_0, dist_1)
        print(distance_3)
        ```

        stdout:
        ```
        0.3575654913778237
        0.15040773967762736
        0.03969123741566945
        0.20105866986400994
        ```

- ### Pre-defined neural networks

    ```python
    from deeplotx import (
        FeedForward, 
        MultiHeadFeedForward, 
        LinearRegression, 
        LogisticRegression, 
        SoftmaxRegression, 
        RecursiveSequential, 
        LongContextRecursiveSequential, 
        RoPE, 
        Attention, 
        MultiHeadAttention, 
        RoFormerEncoder, 
        AutoRegression, 
        LongContextAutoRegression 
    )
    ```

    The fundamental FFN (MLPs):

    ```python
    from typing_extensions import override

    import torch
    from torch import nn

    from deeplotx.nn.base_neural_network import BaseNeuralNetwork


    class FeedForwardUnit(BaseNeuralNetwork):
        def __init__(self, feature_dim: int, expansion_factor: int | float = 2,
                    bias: bool = True, dropout_rate: float = 0.05, model_name: str | None = None,
                    device: str | None = None, dtype: torch.dtype | None = None):
            super().__init__(in_features=feature_dim, out_features=feature_dim, model_name=model_name, device=device, dtype=dtype)
            self._dropout_rate = dropout_rate
            self.up_proj = nn.Linear(in_features=feature_dim, out_features=int(feature_dim * expansion_factor),
                                    bias=bias, device=self.device, dtype=self.dtype)
            self.down_proj = nn.Linear(in_features=int(feature_dim * expansion_factor), out_features=feature_dim,
                                    bias=bias, device=self.device, dtype=self.dtype)
            self.parametric_relu = nn.PReLU(num_parameters=1, init=5e-3,
                                            device=self.device, dtype=self.dtype)
            self.layer_norm = nn.LayerNorm(normalized_shape=self.up_proj.in_features, eps=1e-9,
                                        device=self.device, dtype=self.dtype)

        @override
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
            residual = x
            x = self.layer_norm(x)
            x = self.up_proj(x)
            x = self.parametric_relu(x)
            if self._dropout_rate > .0:
                x = torch.dropout(x, p=self._dropout_rate, train=self.training)
            return self.down_proj(x) + residual


    class FeedForward(BaseNeuralNetwork):
        def __init__(self, feature_dim: int, num_layers: int = 1, expansion_factor: int | float = 2,
                    bias: bool = True, dropout_rate: float = 0.05, model_name: str | None = None,
                    device: str | None = None, dtype: torch.dtype | None = None):
            if num_layers < 1:
                raise ValueError('num_layers cannot be less than 1.')
            super().__init__(in_features=feature_dim, out_features=feature_dim, model_name=model_name, device=device, dtype=dtype)
            self.ffn_layers = nn.ModuleList([FeedForwardUnit(feature_dim=feature_dim,
                                                            expansion_factor=expansion_factor, bias=bias,
                                                            dropout_rate=dropout_rate,
                                                            device=self.device, dtype=self.dtype) for _ in range(num_layers)])

        @override
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
            for ffn in self.ffn_layers:
                x = ffn(x)
            return x
    ```

    Attention:

    ```python
    from typing_extensions import override

    import torch

    from deeplotx.nn.base_neural_network import BaseNeuralNetwork
    from deeplotx.nn.feed_forward import FeedForward
    from deeplotx.nn.rope import RoPE, DEFAULT_THETA


    class Attention(BaseNeuralNetwork):
        def __init__(self, feature_dim: int, bias: bool = True, positional: bool = True,
                    proj_layers: int = 1, proj_expansion_factor: int | float = 1.5, dropout_rate: float = 0.02,
                    model_name: str | None = None, device: str | None = None, dtype: torch.dtype | None = None,
                    **kwargs):
            super().__init__(in_features=feature_dim, out_features=feature_dim, model_name=model_name,
                            device=device, dtype=dtype)
            self._positional = positional
            self._feature_dim = feature_dim
            self.q_proj = FeedForward(feature_dim=self._feature_dim, num_layers=proj_layers,
                                    expansion_factor=proj_expansion_factor,
                                    bias=bias, dropout_rate=dropout_rate, device=self.device, dtype=self.dtype)
            self.k_proj = FeedForward(feature_dim=self._feature_dim, num_layers=proj_layers,
                                    expansion_factor=proj_expansion_factor,
                                    bias=bias, dropout_rate=dropout_rate, device=self.device, dtype=self.dtype)
            self.v_proj = FeedForward(feature_dim=self._feature_dim, num_layers=proj_layers,
                                    expansion_factor=proj_expansion_factor,
                                    bias=bias, dropout_rate=dropout_rate, device=self.device, dtype=self.dtype)
            if self._positional:
                self.rope = RoPE(feature_dim=self._feature_dim, theta=kwargs.get('theta', DEFAULT_THETA),
                                device=self.device, dtype=self.dtype)

        def _attention(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            q, k = self.q_proj(x), self.k_proj(y)
            if self._positional:
                q, k = self.rope(q), self.rope(k)
            attn = torch.matmul(q, k.transpose(-2, -1))
            attn = attn / (self._feature_dim ** 0.5)
            attn = attn.masked_fill(mask == 0, -1e9) if mask is not None else attn
            return torch.softmax(attn, dtype=self.dtype, dim=-1)

        @override
        def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
            x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
            y = x if y is None else self.ensure_device_and_dtype(y, device=self.device, dtype=self.dtype)
            if mask is not None:
                mask = self.ensure_device_and_dtype(mask, device=self.device, dtype=self.dtype)
            v = self.v_proj(y)
            return torch.matmul(self._attention(x, y, mask), v)
    ```

- ### Text binary classification task with predefined trainer

    ```python
    from deeplotx import TextBinaryClassifierTrainer, LongTextEncoder
    from deeplotx.util import get_files, read_file

    long_text_encoder = LongTextEncoder(
        max_length=2048, 
        chunk_size=448, 
        overlapping=32, 
        cache_capacity=512 
    )
    trainer = TextBinaryClassifierTrainer(
        long_text_encoder=long_text_encoder,
        batch_size=2,
        train_ratio=0.9 
    )
    pos_data_path = 'path/to/pos_dir'
    neg_data_path = 'path/to/neg_dir'
    pos_data = [read_file(x) for x in get_files(pos_data_path)]
    neg_data = [read_file(x) for x in get_files(neg_data_path)]
    model = trainer.train(pos_data, neg_data, 
                        num_epochs=36, learning_rate=2e-5, 
                        balancing_dataset=True, alpha=1e-4, 
                        rho=.2, encoder_layers=2, 
                        attn_heads=8, 
                        recursive_layers=2) 
    model.save(model_name='test_model', model_dir='model')
    model = model.load(model_name='test_model', model_dir='model')
    model.predict(long_text_encoder.encode('这是一个测试文本.', flatten=False))
    ```
