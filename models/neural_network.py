from typing import Sequence

import torch
from torch import nn


class PyTorchMLP(nn.Module):
    """Fully connected PyTorch network architecture for regression."""

    def __init__(self, input_size: int, hidden_layers: Sequence[int],
                 activation: str) -> None:
        super(PyTorchMLP, self).__init__()
        activation_layers = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "logistic": nn.Sigmoid
        }
        if activation not in activation_layers:
            raise ValueError("Unsupported activation: {}".format(activation))

        layers = []
        previous_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            # layers.append(activation_layers[activation]())
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(p=0.2))
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X).squeeze(-1)
