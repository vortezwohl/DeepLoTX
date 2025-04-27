import logging
from typing_extensions import override

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from lotc.util.read_file import get_files, read_file
from lotc.embedding.long_text_embedding import long_text_embedding
from lotc.nn.logistic_regression import LogisticRegression
from lotc.trainer.base_trainer import BaseTrainer

logger = logging.getLogger('lotc.trainer')


class FileBinaryClassifierTrainer(BaseTrainer):
    def __init__(self, model_name: str, max_length: int, num_epochs: int,
                 learning_rate: float = 2e-5, batch_size: int = 2,
                 chunk_size: int = 256, train_ratio: float = 0.8,
                 train_loss_threshold: float = 0.0, valid_loss_threshold: float = 0.0):
        super().__init__(model_name=model_name, num_epochs=num_epochs,
                         learning_rate=learning_rate, batch_size=batch_size,
                         train_ratio=train_ratio, train_loss_threshold=train_loss_threshold,
                         valid_loss_threshold=valid_loss_threshold)
        self._max_length = max_length
        self._chunk_size = chunk_size

    @override
    def train(self, positive_file_path: str, negative_file_path: str) -> LogisticRegression:
        pos_files = get_files(positive_file_path)
        neg_files = get_files(negative_file_path)
        min_length = min(len(pos_files), len(neg_files))
        pos_files = pos_files[:min_length]
        neg_files = neg_files[:min_length]
        file_contents = [read_file(p) for p in pos_files + neg_files]
        labels = ([torch.tensor([1.0], dtype=torch.float32) for _ in range(len(pos_files))]
                  + [torch.tensor([0.0], dtype=torch.float32) for _ in range(len(neg_files))])
        text_embeddings = [long_text_embedding(x, max_length=self._max_length, chunk_size=self._chunk_size)
                           for x in file_contents]
        feature_dim = text_embeddings[0][0]
        inputs = torch.stack([x[1] for x in text_embeddings])
        labels = torch.stack(labels)
        dataset_size = len(labels)
        train_size = int(self._train_ratio * dataset_size)
        train_dataset = TensorDataset(inputs[:train_size], labels[:train_size])
        valid_dataset = TensorDataset(inputs[train_size:], labels[train_size:])
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self._batch_size, shuffle=True)
        if self.model is not None and self.model.fc1.in_features != feature_dim:
            logger.warning("The dimension of features doesn't match. A new model instance will be created.")
            self.model = None
        if self.model is None:
            self.model = LogisticRegression(input_dim=feature_dim, train=True)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self._learning_rate)
        for epoch in range(self._num_epochs):
            self.model.train()
            total_loss = 0.0
            for batch_texts, batch_labels in train_loader:
                outputs = self.model(batch_texts)
                loss = loss_function(outputs, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 3 == 0:
                total_valid_loss = 0.0
                for batch_texts, batch_labels in valid_loader:
                    with torch.no_grad():
                        self.model._train = False
                        outputs = self.model(batch_texts)
                        loss = loss_function(outputs, batch_labels)
                        total_valid_loss += loss.item()
                        self.model._train = True
                logger.debug(f"Epoch {epoch + 1}/{self._num_epochs} | "
                             f"Train Loss: {total_loss:.4f} | "
                             f"Valid Loss: {total_valid_loss:.4f}")
                if total_valid_loss < self._valid_loss_threshold:
                    break
            logger.debug(f"Epoch {epoch + 1}/{self._num_epochs} | Train Loss: {total_loss:.4f}")
            if total_loss < self._train_loss_threshold:
                break
        return self.model

    @override
    def save(self) -> LogisticRegression:
        return super().save()

    @override
    def load(self) -> LogisticRegression:
        return super().load()
