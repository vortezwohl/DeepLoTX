import logging
from typing_extensions import override

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from deeplotx.encoder.long_text_encoder import LongTextEncoder
from deeplotx.nn.long_context_recursive_sequential import LongContextRecursiveSequential
from deeplotx.trainer.base_trainer import BaseTrainer

logger = logging.getLogger('deeplotx.trainer')


class TextBinaryClassifierTrainer(BaseTrainer):
    def __init__(self, long_text_encoder: LongTextEncoder, batch_size: int = 2, train_ratio: float = 0.8):
        super().__init__(batch_size=batch_size, train_ratio=train_ratio)
        self._long_text_encoder = long_text_encoder
        self.device = self._long_text_encoder.device
        self.train_dataset_loader = None
        self.valid_dataset_loader = None

    @override
    def train(self, positive_texts: list[str], negative_texts: list[str],
              num_epochs: int, learning_rate: float = 2e-6, balancing_dataset: bool = True,
              train_loss_threshold: float = 0.0, valid_loss_threshold: float = 0.0,
              alpha: float = 1e-4, rho: float = 0.2,
              hidden_dim: int = 256, recursive_layers: int = 2) -> LongContextRecursiveSequential:
        if balancing_dataset:
            min_length = min(len(positive_texts), len(negative_texts))
            positive_texts = positive_texts[:min_length]
            negative_texts = negative_texts[:min_length]
        all_texts = positive_texts + negative_texts
        text_embeddings = [self._long_text_encoder.encode(x, flatten=False) for x in all_texts]
        feature_dim = text_embeddings[0].shape[-1]
        dtype = text_embeddings[0].dtype
        labels = ([torch.tensor([1.], dtype=dtype, device=self.device) for _ in range(len(positive_texts))]
                  + [torch.tensor([.0], dtype=dtype, device=self.device) for _ in range(len(negative_texts))])
        inputs = torch.stack(text_embeddings).to(self.device)
        labels = torch.stack(labels).to(self.device)
        dataset_size = len(labels)
        train_size = int(self._train_ratio * dataset_size)
        train_dataset = TensorDataset(inputs[:train_size], labels[:train_size])
        valid_dataset = TensorDataset(inputs[train_size:], labels[train_size:])
        self.train_dataset_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        self.valid_dataset_loader = DataLoader(valid_dataset, batch_size=self._batch_size, shuffle=True)

        if self.model is not None and self.model.fc1.in_features != feature_dim:
            logger.warning("The dimension of features doesn't match. A new model instance will be created.")
            self.model = None
        if self.model is None:
            self.model = LongContextRecursiveSequential(input_dim=feature_dim, output_dim=1,
                                                        hidden_dim=hidden_dim,
                                                        recursive_layers=recursive_layers,
                                                        device=self.device, dtype=dtype)
        loss_function = nn.BCELoss()
        optimizer = optim.Adamax(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            for batch_texts, batch_labels in self.train_dataset_loader:
                outputs = torch.sigmoid(self.model.forward(batch_texts, self.model.initial_state(batch_texts.shape[0]))[0])
                loss = loss_function(outputs, batch_labels) + self.model.elastic_net(alpha=alpha, rho=rho)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 3 == 0:
                total_valid_loss = 0.0
                for batch_texts, batch_labels in self.valid_dataset_loader:
                    with torch.no_grad():
                        self.model.eval()
                        outputs = torch.sigmoid(self.model.forward(batch_texts, self.model.initial_state(batch_texts.shape[0]))[0])
                        loss = loss_function(outputs, batch_labels) + self.model.elastic_net(alpha=alpha, rho=rho)
                        total_valid_loss += loss.item()
                        self.model.train()
                logger.debug(f"Epoch {epoch + 1}/{num_epochs} | "
                             f"Train Loss: {total_loss:.4f} | "
                             f"Valid Loss: {total_valid_loss:.4f}")
                if total_valid_loss < valid_loss_threshold:
                    break
            else:
                logger.debug(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {total_loss:.4f}")
            if total_loss < train_loss_threshold:
                break
        return self.model
