import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _PyTorchMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, activation):
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

    def forward(self, X):
        return self.network(X).squeeze(-1)


class NeuralNetworkPredictor:
    def __init__(self, hidden_layers=(64, 32), activation="relu", max_iter=500,
                 random_seed=42, validation_fraction=0.15, n_iter_no_change=20,
                 batch_size=64, learning_rate=0.001):
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scaler = None
        self.model = None
        self.iterations_run = 0

    def fit(self, X, y):
        if self.activation not in ("relu", "tanh", "logistic"):
            raise ValueError("Unsupported activation: {}".format(self.activation))

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_validation, y_train, y_validation = train_test_split(
            X_scaled,
            np.asarray(y, dtype=np.float32),
            test_size=self.validation_fraction,
            random_state=self.random_seed,
            shuffle=True
        )

        train_dataset = TensorDataset(
            torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_train, dtype=np.float32))
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_seed)
        )
        X_validation_tensor = torch.from_numpy(np.asarray(X_validation, dtype=np.float32))
        y_validation_tensor = torch.from_numpy(np.asarray(y_validation, dtype=np.float32))

        device = torch.device("cpu")
        self.model = _PyTorchMLP(X.shape[1], self.hidden_layers, self.activation).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_function = nn.MSELoss()

        best_loss = float("inf")
        no_improvement = 0
        best_state = None

        self.model.train()
        for epoch in range(self.max_iter):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X.to(device))
                loss = loss_function(predictions, batch_y.to(device))
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                validation_predictions = self.model(X_validation_tensor.to(device))
                validation_loss = float(loss_function(
                    validation_predictions,
                    y_validation_tensor.to(device)
                ).item())
            self.model.train()

            self.iterations_run = epoch + 1
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                no_improvement = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                no_improvement += 1

            if no_improvement >= self.n_iter_no_change:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, X):
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been fitted")
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.from_numpy(np.asarray(X_scaled, dtype=np.float32))
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
        return predictions

    def details(self):
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
