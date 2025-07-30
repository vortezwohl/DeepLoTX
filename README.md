[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/vortezwohl/DeepLoTX)

# Deep Long Text Learning Kit

> Author: 吴子豪

**开箱即用的长文本语义建模框架**

## 安装

- 使用 pip

    ```
    pip install -U deeplotx
    ```

- 使用 uv (推荐)

    ```
    uv add -U deeplotx
    ```
  
- 从 github 安装最新特性

    ```
    pip install -U git+https://github.com/vortezwohl/DeepLoTX.git
    ```

## 核心功能

- ### 长文本嵌入

    - **基于通用 BERT 的长文本嵌入** (最大支持长度, 无限长, 可通过 max_length 限制长度)

        ```python
        from deeplotx import LongTextEncoder

        # 块大小为 448 个 tokens, 块间重叠部分为 32 个 tokens.
        encoder = LongTextEncoder(
            chunk_size=448,
            overlapping=32
        )
        # 对 "我是吴子豪, 这是一个测试文本." 计算嵌入, 并堆叠.
        encoder.encode('我是吴子豪, 这是一个测试文本.', flatten=False)
        ```

        输出:
        ```
        tensor([ 2.2316e-01,  2.0300e-01,  ...,  1.5578e-01, -6.6735e-02])
        ```

    - **基于 Longformer 的长文本嵌入** (最大支持长度 4096 个 tokens)

        ```python
        from deeplotx import LongformerEncoder

        encoder = LongformerEncoder()
        encoder.encode('我是吴子豪, 这是一个测试文本.')
        ```

        输出:
        ```
        tensor([-2.7490e-02,  6.6503e-02, ..., -6.5937e-02,  6.7802e-03])
        ```

- ### 相似性计算

    - **基于向量的相似性**

        ```python
        import deeplotx.similarity as sim

        vector_0, vector_1 = [1, 2, 3, 4], [4, 3, 2, 1]
        # 欧几里得距离
        distance_0 = sim.euclidean_similarity(vector_0, vector_1)
        print(distance_0)
        # 余弦距离
        distance_1 = sim.cosine_similarity(vector_0, vector_1)
        print(distance_1)
        # 切比雪夫距离
        distance_2 = sim.chebyshev_similarity(vector_0, vector_1)
        print(distance_2)
        ```

        输出:
        ```
        4.47213595499958
        0.33333333333333337
        3
        ```

    - **基于集合的相似性**

        ```python
        import deeplotx.similarity as sim

        set_0, set_1 = {1, 2, 3, 4}, {4, 5, 6, 7}
        # 杰卡德距离
        distance_0 = sim.jaccard_similarity(set_0, set_1)
        print(distance_0)
        # Ochiai 距离
        distance_1 = sim.ochiai_similarity(set_0, set_1)
        print(distance_1)
        # Dice 系数
        distance_2 = sim.dice_coefficient(set_0, set_1)
        print(distance_2)
        # Overlap 系数
        distance_3 = sim.overlap_coefficient(set_0, set_1)
        print(distance_3)
        ```

        输出:
        ```
        0.1428571428572653
        0.2500000000001875
        0.25000000000009376
        0.2500000000001875
        ```

    - **基于概率分布的相似性**

        ```python
        import deeplotx.similarity as sim

        dist_0, dist_1 = [0.3, 0.2, 0.1, 0.4], [0.2, 0.1, 0.3, 0.4]
        # 交叉熵
        distance_0 = sim.cross_entropy(dist_0, dist_1)
        print(distance_0)
        # KL 散度
        distance_1 = sim.kl_divergence(dist_0, dist_1)
        print(distance_1)
        # JS 散度
        distance_2 = sim.js_divergence(dist_0, dist_1)
        print(distance_2)
        # Hellinger 距离
        distance_3 = sim.hellinger_distance(dist_0, dist_1)
        print(distance_3)
        ```

        输出:
        ```
        0.3575654913778237
        0.15040773967762736
        0.03969123741566945
        0.20105866986400994
        ```

- ### 预定义深度神经网络

    ```python
    from deeplotx import (
        FeedForward,  # 前馈神经网络
        LinearRegression,  # 线性回归
        LogisticRegression,  # 逻辑回归 / 二分类 / 多标签分类
        SoftmaxRegression,  # Softmax 回归 / 多分类
        RecursiveSequential,  # 序列模型 / 循环神经网络
        LongContextRecursiveSequential,  # 长上下文序列模型 / 自注意力融合循环神经网络
        RoPE,  # RoPE 位置编码
        Attention,  # 自注意力 / 交叉注意力
        MultiHeadAttention,  # 并行多头注意力
        RoFormerEncoder,  # Roformer (Transformer + RoPE) 编码器模型
        AutoRegression,  # 自回归模型 / 循环神经网络
        LongContextAutoRegression  # 长上下文自回归模型 / 自注意力融合循环神经网络
    )
    ```

    基础网络结构:

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

    注意力模块:

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

- ### 使用预定义训练器实现文本二分类任务

    ```python
    from deeplotx import TextBinaryClassifierTrainer, LongTextEncoder
    from deeplotx.util import get_files, read_file

    # 定义向量编码策略 (默认使用 FacebookAI/xlm-roberta-base 作为嵌入模型)
    long_text_encoder = LongTextEncoder(
        max_length=2048,  # 最大文本大小, 超出截断
        chunk_size=448,  # 块大小 (按 Token 计)
        overlapping=32,  # 块间重叠大小 (按 Token 计)
        cache_capacity=512  # 缓存大小
    )

    trainer = TextBinaryClassifierTrainer(
        long_text_encoder=long_text_encoder,
        batch_size=2,
        train_ratio=0.9  # 训练集和验证集比例
    )

    # 读取数据
    pos_data_path = 'path/to/pos_dir'
    neg_data_path = 'path/to/neg_dir'
    pos_data = [read_file(x) for x in get_files(pos_data_path)]
    neg_data = [read_file(x) for x in get_files(neg_data_path)]

    # 开始训练
    model = trainer.train(pos_data, neg_data, 
                        num_epochs=36, learning_rate=2e-5, 
                        balancing_dataset=True, alpha=1e-4, 
                        rho=.2, encoder_layers=2,  # 2 层 Roformer 编码器
                        attn_heads=8,  # 8 个注意力头
                        recursive_layers=2)  # 2 层 Bi-LSTM

    # 保存模型权重
    model.save(model_name='test_model', model_dir='model')

    # 加载已保存的模型
    model = model.load(model_name='test_model', model_dir='model')

    # 使用训练好的模型进行预测
    model.predict(long_text_encoder.encode('这是一个测试文本.', flatten=False))
    ```
