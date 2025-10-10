from typing_extensions import override

import torch
from torch import nn

from deeplotx.nn.base_neural_network import BaseNeuralNetwork


class LoRA(BaseNeuralNetwork):
    def __init__(self, input_dim: int, output_dim: int, rank: int = 8, alpha: int = 16,
                 dropout_rate: float = .0, model_name: str | None = None, device: str | torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__(in_features=input_dim, out_features=output_dim, model_name=model_name,
                         device=device, dtype=dtype)
        self._rank = rank
        self._alpha = alpha
        self._scaling = self._alpha / self._rank
        self._dropout = nn.Dropout(p=dropout_rate) if dropout_rate > .0 else nn.Identity()
        self.lora_A = nn.Linear(in_features=input_dim, out_features=rank, bias=False,
                                device=self.device, dtype=self.dtype)
        self.lora_B = nn.Linear(in_features=rank, out_features=output_dim, bias=False,
                                device=self.device, dtype=self.dtype)
        nn.init.normal_(self.lora_A.weight, mean=.0, std=.01)
        nn.init.zeros_(self.lora_B.weight)
        self.w0 = None

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.w0, nn.Module):
            raise ValueError('LoRA adapter was not mounted successfully.')
        original_out = self.w0(x)
        lora_out = self.lora_B(self._dropout(self.lora_A(x))) * self._scaling
        return original_out + lora_out

    @staticmethod
    def apply_to(model: nn.Module, target_modules: list[str] | str, rank: int = 8, alpha: int = 16,
                 dropout_rate: float = .0) -> nn.Module:
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        for layer_name, module in model.named_modules():
            if any(_name in layer_name.split('.')[-1] for _name in target_modules):
                lora = LoRA(input_dim=module.in_features, output_dim=module.out_features,
                            rank=rank, alpha=alpha, dropout_rate=dropout_rate,
                            device=next(module.parameters()).device,
                            dtype=next(module.parameters()).dtype)
                lora.w0 = module
                parent_name = layer_name.rsplit('.', 1)[0] if '.' in layer_name else ''
                child_name = layer_name.split('.')[-1]
                parent_module = dict(model.named_modules())[parent_name] if parent_name else model
                setattr(parent_module, child_name, lora)
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'lora_A.weight' in name or 'lora_B.weight' in name:
                param.requires_grad = True
        return model
