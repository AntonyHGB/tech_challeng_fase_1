import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from churn_predictor.models.dataset import TabularDataset

LOGGER = logging.getLogger(__name__)


class ChurnMLP(nn.Module):
    """Arquitetura de Multi-Layer Perceptron para classificação de Churn."""

    def __init__(self, input_dim: int, hidden_layers: list[int], dropout_rate: float = 0.2):
        """Inicializa a arquitetura MLP."""
        super().__init__()

        layers = []
        in_features = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim

        # Camada de saída (binária, sem ativação, pois usaremos BCEWithLogitsLoss)
        layers.append(nn.Linear(in_features, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa forward pass pela rede."""
        return self.network(x)


class PyTorchMLPWrapper:
    """Wrapper compatível com scikit-learn (possui fit e predict_proba).

    Implementa Early Stopping internamente.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 10,
        device: str = "cpu",
    ):
        """Inicializa o wrapper com hiperparâmetros de treino."""
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        # MPS não funciona bem com tipos incompatíveis para dados tabulares.
        # Caso CUDA esteja disponível, usamos CUDA.
        self.device = (
            "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        )

        self.model = ChurnMLP(input_dim, hidden_layers, dropout_rate).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.best_state_dict = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """Treina a MLP com early stopping baseado na validation loss."""
        train_dataset = TabularDataset(x_train, y_train)
        val_dataset = TabularDataset(x_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0

        LOGGER.info(f"Starting PyTorch training on {self.device}...")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * batch_x.size(0)

            train_loss /= len(train_dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)

            val_loss /= len(val_dataset)

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                LOGGER.info(
                    f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}"
                )
                break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Retorna probabilidades por classe.

        Formato: [prob_classe_0, prob_classe_1], compatível com scikit-learn.
        """
        self.model.eval()
        dataset = TabularDataset(x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probs_1 = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                # Aplicamos sigmoid para obter a probabilidade da classe 1
                batch_probs = torch.sigmoid(logits).cpu().numpy()
                probs_1.extend(batch_probs)

        probs_1 = np.array(probs_1).flatten()
        # Constrói array 2D para as duas classes
        probs_0 = 1.0 - probs_1
        return np.column_stack((probs_0, probs_1))
