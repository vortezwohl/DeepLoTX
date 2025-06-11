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

    - **基于通用 BERT 的长文本嵌入** (最大支持长度, 无限长, 通过 max_length 定义)

        ```python
        from deeplotx import LongTextEncoder

        # 最大文本长度为 2048 个 tokens, 块大小为 512 个 tokens, 块间重叠部分为 64 个 tokens.
        encoder = LongTextEncoder(
            max_length=2048,
            chunk_size=512,
            overlapping=64
        )
        # 对 "我是吴子豪, 这是一个测试文本." 计算嵌入, 并展平.
        encoder.encode('我是吴子豪, 这是一个测试文本.', flatten=True, use_cache=True)
        ```

        输出:
        ```
        tensor([ 0.5163,  0.2497,  0.5896,  ..., -0.9815, -0.3095,  0.4232])
        ```

    - **基于 Longformer 的长文本嵌入** (最大支持长度 4096 个 tokens)

        ```python
        from deeplotx import LongformerEncoder

        encoder = LongformerEncoder()
        encoder.encode('我是吴子豪, 这是一个测试文本.')
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
        LinearRegression,  # 线性回归
        LogisticRegression,  # 逻辑回归 / 二分类 / 多标签分类
        SoftmaxRegression,  # Softmax 回归 / 多分类
        RecursiveSequential,  # 序列模型 / 循环神经网络
        AutoRegression  # 自回归模型
    )
    ```

    基础网络结构:

    ```python
    from typing_extensions import override

    import torch
    from torch import nn

    from deeplotx.nn.base_neural_network import BaseNeuralNetwork


    class LinearRegression(BaseNeuralNetwork):
        def __init__(self, input_dim: int, output_dim: int, model_name: str | None = None):
            super().__init__(model_name=model_name)
            self.fc1 = nn.Linear(input_dim, 1024)
            self.fc1_to_fc4_res = nn.Linear(1024, 64)
            self.fc2 = nn.Linear(1024, 768)
            self.fc3 = nn.Linear(768, 128)
            self.fc4 = nn.Linear(128, 64)
            self.fc5 = nn.Linear(64, output_dim)
            self.parametric_relu_1 = nn.PReLU(num_parameters=1, init=5e-3)
            self.parametric_relu_2 = nn.PReLU(num_parameters=1, init=5e-3)
            self.parametric_relu_3 = nn.PReLU(num_parameters=1, init=5e-3)
            self.parametric_relu_4 = nn.PReLU(num_parameters=1, init=5e-3)

        @override
        def forward(self, x) -> torch.Tensor:
            fc1_out = self.parametric_relu_1(self.fc1(x))
            x = nn.LayerNorm(normalized_shape=1024, eps=1e-9)(fc1_out)
            x = torch.dropout(x, p=0.2, train=self.training)
            x = self.parametric_relu_2(self.fc2(x))
            x = nn.LayerNorm(normalized_shape=768, eps=1e-9)(x)
            x = torch.dropout(x, p=0.2, train=self.training)
            x = self.parametric_relu_3(self.fc3(x))
            x = torch.dropout(x, p=0.2, train=self.training)
            x = self.parametric_relu_4(self.fc4(x)) + self.fc1_to_fc4_res(fc1_out)
            x = self.fc5(x)
            return x
    ```
