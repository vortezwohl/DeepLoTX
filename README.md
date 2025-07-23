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
        LongContextRecursiveSequential,  # 长上下文序列模型 / 自注意力融合循环神经网络
        SelfAttention,  # 自注意力模块
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
    
    
    class LinearRegression(BaseNeuralNetwork):
        def __init__(self, input_dim: int, output_dim: int, model_name: str | None = None,
                     device: str | None = None, dtype: torch.dtype | None = None):
            super().__init__(model_name=model_name, device=device, dtype=dtype)
            self.fc1 = nn.Linear(input_dim, 1024, device=self.device, dtype=self.dtype)
            self.fc1_to_fc4_res = nn.Linear(1024, 64, device=self.device, dtype=self.dtype)
            self.fc2 = nn.Linear(1024, 768, device=self.device, dtype=self.dtype)
            self.fc3 = nn.Linear(768, 128, device=self.device, dtype=self.dtype)
            self.fc4 = nn.Linear(128, 64, device=self.device, dtype=self.dtype)
            self.fc5 = nn.Linear(64, output_dim, device=self.device, dtype=self.dtype)
            self.parametric_relu_1 = nn.PReLU(num_parameters=1, init=5e-3, device=self.device, dtype=self.dtype)
            self.parametric_relu_2 = nn.PReLU(num_parameters=1, init=5e-3, device=self.device, dtype=self.dtype)
            self.parametric_relu_3 = nn.PReLU(num_parameters=1, init=5e-3, device=self.device, dtype=self.dtype)
            self.parametric_relu_4 = nn.PReLU(num_parameters=1, init=5e-3, device=self.device, dtype=self.dtype)
    
        @override
        def forward(self, x) -> torch.Tensor:
            x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
            fc1_out = self.parametric_relu_1(self.fc1(x))
            x = nn.LayerNorm(normalized_shape=1024, eps=1e-9, device=self.device, dtype=self.dtype)(fc1_out)
            x = torch.dropout(x, p=0.2, train=self.training)
            x = self.parametric_relu_2(self.fc2(x))
            x = nn.LayerNorm(normalized_shape=768, eps=1e-9, device=self.device, dtype=self.dtype)(x)
            x = torch.dropout(x, p=0.2, train=self.training)
            x = self.parametric_relu_3(self.fc3(x))
            x = torch.dropout(x, p=0.2, train=self.training)
            x = self.parametric_relu_4(self.fc4(x)) + self.fc1_to_fc4_res(fc1_out)
            x = self.fc5(x)
            return x
    ```

    自注意力模块:

    ```python
    from typing_extensions import override

    import torch
    from torch import nn, softmax

    from deeplotx.nn.base_neural_network import BaseNeuralNetwork


    class SelfAttention(BaseNeuralNetwork):
        def __init__(self, feature_dim: int, model_name: str | None = None,
                    device: str | None = None, dtype: torch.dtype | None = None):
            super().__init__(model_name=model_name, device=device, dtype=dtype)
            self._feature_dim = feature_dim
            self.q_proj = nn.Linear(in_features=self._feature_dim, out_features=self._feature_dim,
                                    bias=True, device=self.device, dtype=self.dtype)
            self.k_proj = nn.Linear(in_features=self._feature_dim, out_features=self._feature_dim,
                                    bias=True, device=self.device, dtype=self.dtype)
            self.v_proj = nn.Linear(in_features=self._feature_dim, out_features=self._feature_dim,
                                    bias=True, device=self.device, dtype=self.dtype)

        def _attention(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            q, k = self.q_proj(x), self.k_proj(x)
            attn = torch.matmul(q, k.transpose(-2, -1))
            attn = attn / (self._feature_dim ** 0.5)
            attn = attn.masked_fill(mask == 0, -1e9) if mask is not None else attn
            return softmax(attn, dim=-1)

        @override
        def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            x = self.ensure_device_and_dtype(x, device=self.device, dtype=self.dtype)
            if mask is not None:
                mask = self.ensure_device_and_dtype(mask, device=self.device, dtype=self.dtype)
            v = self.v_proj(x)
            return torch.matmul(self._attention(x, mask), v)
    ```

- ### 使用预定义训练器实现文本二分类任务

    ```python
    from deeplotx import TextBinaryClassifierTrainer, LongTextEncoder
    from deeplotx.util import get_files, read_file

    # 定义向量编码策略 (默认使用 bert-base-uncased 作为嵌入模型)
    long_text_encoder = LongTextEncoder(
        max_length=2048,  # 最大文本大小, 超出截断
        chunk_size=448,  # 块大小 (按 Token 计)
        overlapping=32  # 块间重叠大小 (按 Token 计)
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
                          num_epochs=36, learning_rate=2e-5,  # 设置训练轮数和学习率
                          balancing_dataset=True,  # 是否平衡数据集
                          alpha=1e-4, rho=.2,  # 设置 elastic net 正则化的超参数 alpha 和 rho
                          hidden_dim=256, recursive_layers=2)  # 设置循环神经网络的结构

    # 保存模型权重
    model.save(model_name='test_model', model_dir='model')

    # 加载已保存的模型
    model = model.load(model_name='test_model', model_dir='model')

    # 使用训练好的模型进行预测
    model.predict(long_text_encoder.encode('这是一个测试文本.', flatten=False))
    ```
