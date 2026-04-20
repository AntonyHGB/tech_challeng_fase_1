import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """Dataset PyTorch genérico para dados tabulares numéricos.

    Recebe arrays NumPy (típicos de saída de ColumnTransformer)
    e os converte para tensores PyTorch.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray | None = None) -> None:
        """Inicializa o dataset.

        Args:
            x: Matriz de features.
            y: Vetor de labels (opcional, para inferência).
        """
        self.x = torch.tensor(x, dtype=torch.float32)

        if y is not None:
            # BCEWithLogitsLoss espera floats como target
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None

    def __len__(self) -> int:
        """Retorna o número de amostras."""
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Retorna uma amostra específica."""
        if self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx]
