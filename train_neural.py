#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
from typing import List

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset_utils import load_dataset, split_dataset
from models.neural_network import NeuralNetworkPredictor


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
    model = NeuralNetworkPredictor(
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        max_iter=args.max_iter,
        random_seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions))
    }

    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.output_dir, "neural_model.joblib"))
    results = {
        "model_type": "neural",
        "framework": "PyTorch",
        "dataset": os.path.abspath(args.data),
        "random_seed": args.seed,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": metrics,
        "model_details": model.details()
    }
    with open(os.path.join(args.output_dir, "neural_results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print("Test metrics: MAE={:.6f}, RMSE={:.6f}, R2={:.6f}".format(
        metrics["MAE"], metrics["RMSE"], metrics["R2"]))
    print("Saved PyTorch neural results to {}".format(os.path.abspath(args.output_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
