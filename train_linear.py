#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset_utils import load_dataset, split_dataset
from models.linear_regression import LinearRegressionPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the linear regression model")
    parser.add_argument("--data", default=os.path.join("dataset", "industrial_dataset.csv"))
    parser.add_argument("--target", default="Target")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="output-linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--no-fit-intercept", action="store_true")
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

    print("Training linear regression model")
    model = LinearRegressionPredictor(
        fit_intercept=not args.no_fit_intercept
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions))
    }

    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.output_dir, "linear_model.joblib"))
    results = {
        "model_type": "linear-regression",
        "framework": "scikit-learn",
        "dataset": os.path.abspath(args.data),
        "random_seed": args.seed,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": metrics,
        "model_details": model.details()
    }
    with open(os.path.join(args.output_dir, "linear_results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print("Test metrics: MAE={:.6f}, RMSE={:.6f}, R2={:.6f}".format(
        metrics["MAE"], metrics["RMSE"], metrics["R2"]))
    print("Saved linear regression results to {}".format(os.path.abspath(args.output_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
