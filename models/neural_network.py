from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _PyTorchMLP(nn.Module):
    def __init__(self, input_size: int, hidden_layers: Sequence[int],
                 activation: str) -> None:
        super(_PyTorchMLP, self).__init__()
        layers = []
        previous_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "logistic":
                layers.append(nn.Sigmoid())
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X).squeeze(-1)


class NeuralNetworkPredictor:
    def __init__(self, hidden_layers: Sequence[int] = (64, 32),
                 activation: str = "relu", max_iter: int = 500,
                 random_seed: int = 42, batch_size: int = 64,
                 learning_rate: float = 0.001) -> None:
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[nn.Module] = None
        self.iterations_run = 0

    def fit(self, X: pd.DataFrame,
            y: pd.Series) -> "NeuralNetworkPredictor":
        if self.activation not in ("relu", "tanh", "logistic"):
            raise ValueError("Unsupported activation: {}".format(self.activation))

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.scaler = StandardScaler()
        X_scaled = np.asarray(self.scaler.fit_transform(X), dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32)

        dataset = TensorDataset(
            torch.from_numpy(X_scaled),
            torch.from_numpy(y_array)
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_seed)
        )

        self.model = _PyTorchMLP(X.shape[1], self.hidden_layers, self.activation)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_function = nn.MSELoss()

        self.model.train()
        for epoch in range(self.max_iter):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = loss_function(predictions, batch_y)
                loss.backward()
                optimizer.step()
            self.iterations_run = epoch + 1

        self.model.eval()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been fitted")
        X_scaled = np.asarray(self.scaler.transform(X), dtype=np.float32)
        X_tensor = torch.from_numpy(X_scaled)
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
        return predictions

    def details(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        return {
            "framework": "PyTorch",
            "hidden_layer_sizes": list(self.hidden_layers),
            "activation": self.activation,
            "max_iter": self.max_iter,
            "iterations_run": self.iterations_run,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate
        }
