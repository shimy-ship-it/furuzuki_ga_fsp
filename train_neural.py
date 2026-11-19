#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset_utils import load_dataset, split_dataset
from models.neural_network import PyTorchMLP


def train_model(X: pd.DataFrame, y: pd.Series, hidden_layers: Sequence[int],
                activation: str, max_iter: int, random_seed: int,
                batch_size: int, learning_rate: float) -> Tuple[
                    PyTorchMLP, StandardScaler, int]:
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    scaler = StandardScaler()
    X_scaled = np.asarray(scaler.fit_transform(X), dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32)
    dataset = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y_array))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(random_seed)
    )

    model = PyTorchMLP(X.shape[1], hidden_layers, activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    model.train()
    for _ in range(max_iter):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model, scaler, max_iter


def predict(model: PyTorchMLP, scaler: StandardScaler,
            X: pd.DataFrame) -> np.ndarray:
    X_scaled = np.asarray(scaler.transform(X), dtype=np.float32)
    X_tensor = torch.from_numpy(X_scaled)
    model.eval()
    with torch.no_grad():
        return model(X_tensor).numpy()


def parse_hidden_layers(value: str) -> List[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
        if not result or any(size <= 0 for size in result):
            raise ValueError
        return result
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Hidden layers must be positive integers separated by commas, e.g. 64,32")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the PyTorch neural-network model")
    parser.add_argument("--data", default=os.path.join("dataset", "industrial_dataset.csv"))
    parser.add_argument("--target", default="Target")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="output-neural")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--hidden-layers", type=parse_hidden_layers, default=[64, 32])
    parser.add_argument("--activation", choices=["relu", "tanh", "logistic"], default="relu")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.05 <= args.test_size < 0.95:
        raise ValueError("--test-size must be between 0.05 and 0.95")

    np.random.seed(args.seed)
    random.seed(args.seed)
    features, target = load_dataset(args.data, args.target)
    X_train, X_test, y_train, y_test = split_dataset(
        features, target, dataset_dir=args.dataset_dir,
        test_size=args.test_size, random_seed=args.seed
    )

    print("Training PyTorch neural network model")
    model, scaler, iterations_run = train_model(
        X_train,
        y_train,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        max_iter=args.max_iter,
        random_seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    predictions = predict(model, scaler, X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions))
    }

    os.makedirs(args.output_dir, exist_ok=True)
    model_details = {
        "framework": "PyTorch",
        "hidden_layer_sizes": list(args.hidden_layers),
        "activation": args.activation,
        "max_iter": args.max_iter,
        "iterations_run": iterations_run,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate
    }
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "model_details": model_details
    }, os.path.join(args.output_dir, "neural_model.joblib"))
    results = {
        "model_type": "neural",
        "framework": "PyTorch",
        "dataset": os.path.abspath(args.data),
        "random_seed": args.seed,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": metrics,
        "model_details": model_details
    }
    with open(os.path.join(args.output_dir, "neural_results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print("Test metrics: MAE={:.6f}, RMSE={:.6f}, R2={:.6f}".format(
        metrics["MAE"], metrics["RMSE"], metrics["R2"]))
    print("Saved PyTorch neural results to {}".format(os.path.abspath(args.output_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
