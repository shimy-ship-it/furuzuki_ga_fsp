import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class GeneticSVMPredictor:
    def __init__(self, population_size: int = 10, generations: int = 5,
                 cv_folds: int = 3, random_seed: int = 42,
                 tournament_size: int = 3) -> None:
        self.population_size = population_size
        self.generations = generations
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.tournament_size = tournament_size
        self.model = None
        self.selected_features = []
        self._details = {}

    def _parameters(self, chromosome: np.ndarray, n_features: int) -> Dict[str, Any]:
        mask = chromosome[:n_features] > 0.5
        if mask.sum() == 0:
            mask[np.argmax(chromosome[:n_features])] = True
        return {
            "mask": mask,
            "C": float(10.0 ** chromosome[n_features]),
            "epsilon": float(10.0 ** chromosome[n_features + 1]),
            "gamma": "scale" if chromosome[n_features + 2] < 0.5 else "auto"
        }

    def _make_chromosome(self, n_features: int) -> np.ndarray:
        mask = np.random.random(n_features)
        mask[mask < 0.7] = 0.0
        mask[mask >= 0.7] = 1.0
        if mask.sum() == 0:
            mask[np.random.randint(n_features)] = 1.0
        return np.concatenate([
            mask,
            np.array([
                np.random.uniform(-2.0, 3.0),
                np.random.uniform(-3.0, 0.0),
                np.random.random()
            ])
        ])

    def _fitness(self, chromosome: np.ndarray, X: pd.DataFrame,
                 y: pd.Series, n_features: int, cv: KFold) -> float:
        params = self._parameters(chromosome, n_features)
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=params["C"], epsilon=params["epsilon"], gamma=params["gamma"]))
        ])
        scores = cross_val_score(
            pipeline, X.iloc[:, params["mask"]], y, cv=cv,
            scoring="neg_mean_squared_error", n_jobs=1
        )
        return float(np.mean(scores))

    def _tournament(self, population: List[np.ndarray],
                    fitness: List[float]) -> np.ndarray:
        tournament_size = min(self.tournament_size, len(population))
        candidates = np.random.choice(len(population), size=tournament_size, replace=False)
        best = candidates[np.argmax([fitness[i] for i in candidates])]
        return population[best].copy()

    def _crossover(self, parent_a: np.ndarray,
                   parent_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        point = np.random.randint(1, len(parent_a))
        return (np.concatenate([parent_a[:point], parent_b[point:]]),
                np.concatenate([parent_b[:point], parent_a[point:]]))

    def _mutate(self, chromosome: np.ndarray,
                n_features: int) -> np.ndarray:
        chromosome = chromosome.copy()
        for index in range(n_features):
            if np.random.random() < 0.08:
                chromosome[index] = 1.0 - chromosome[index]
        for index in range(n_features, len(chromosome)):
            if np.random.random() < 0.20:
                if index == n_features:
                    chromosome[index] = np.clip(np.random.normal(chromosome[index], 0.5), -2.0, 3.0)
                elif index == n_features + 1:
                    chromosome[index] = np.clip(np.random.normal(chromosome[index], 0.3), -3.0, 0.0)
                else:
                    chromosome[index] = np.clip(np.random.normal(chromosome[index], 0.15), 0.0, 1.0)
        return chromosome

    def fit(self, X: pd.DataFrame,
            y: pd.Series) -> "GeneticSVMPredictor":
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        n_features = X.shape[1]
        if len(y) < 10:
            raise ValueError("Genetic algorithm requires at least 10 training rows")

        cv = KFold(n_splits=min(self.cv_folds, len(y)), shuffle=True, random_state=self.random_seed)
        population = [self._make_chromosome(n_features) for _ in range(self.population_size)]
        fitness = [self._fitness(chromosome, X, y, n_features, cv) for chromosome in population]

        for generation in range(self.generations):
            next_population = []
            elite_index = int(np.argmax(fitness))
            next_population.append(population[elite_index].copy())
            while len(next_population) < self.population_size:
                parent_a = self._tournament(population, fitness)
                parent_b = self._tournament(population, fitness)
                child_a, child_b = self._crossover(parent_a, parent_b)
                next_population.extend([self._mutate(child_a, n_features),
                                        self._mutate(child_b, n_features)])

            population = next_population[:self.population_size]
            fitness = [self._fitness(chromosome, X, y, n_features, cv)
                       for chromosome in population]
            best_index = int(np.argmax(fitness))
            params = self._parameters(population[best_index], n_features)
            rmse = float(np.sqrt(-fitness[best_index]))
            print("GA generation {}/{} - CV RMSE: {:.6f} - features: {}".format(
                generation + 1, self.generations, rmse, int(params["mask"].sum())))

        best_index = int(np.argmax(fitness))
        params = self._parameters(population[best_index], n_features)
        self.selected_features = X.columns[params["mask"]].tolist()

        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=params["C"], epsilon=params["epsilon"], gamma=params["gamma"]))
        ])
        self.model.fit(X[self.selected_features], y)
        self._details = {
            "selected_features": self.selected_features,
            "feature_count": len(self.selected_features),
            "C": params["C"],
            "epsilon": params["epsilon"],
            "gamma": params["gamma"],
            "cv_rmse": float(np.sqrt(-fitness[best_index]))
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        return self.model.predict(X[self.selected_features])

    def details(self) -> Dict[str, Any]:
        return self._details.copy()
