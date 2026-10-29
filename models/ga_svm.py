import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class GeneticSVMPredictor:
    """Binary-encoded genetic algorithm for feature selection with SVM regressor."""

    def __init__(self, population_size: int = 20, generations: int = 30,
                 cv_folds: int = 3, random_seed: int = 42,
                 tournament_size: int = 3, C: float = 1.0,
                 epsilon: float = 0.1, gamma: str = "scale") -> None:
        self.population_size = population_size
        self.generations = generations
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.tournament_size = tournament_size
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = None
        self.selected_features = []
        self._details = {}

    def _make_chromosome(self, n_features: int) -> np.ndarray:
        chromosome = np.random.randint(0, 2, size=n_features, dtype=np.int8)
        if chromosome.sum() == 0:
            chromosome[np.random.randint(0, n_features)] = 1
        return chromosome

    def _selected_mask(self, chromosome: np.ndarray) -> np.ndarray:
        mask = chromosome.astype(bool)
        if mask.sum() == 0:
            mask[0] = True
        return mask

    def _fitness(self, chromosome: np.ndarray, X: pd.DataFrame,
                 y: pd.Series, cv: KFold) -> float:
        mask = self._selected_mask(chromosome)
        if mask.sum() == 0:
            return float("-inf")

        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=self.C, epsilon=self.epsilon, gamma=self.gamma))
        ])
        scores = cross_val_score(
            pipeline,
            X.iloc[:, mask],
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        return float(np.mean(scores))

    def _tournament(self, population: List[np.ndarray],
                    fitness: List[float]) -> np.ndarray:
        tournament_size = min(self.tournament_size, len(population))
        candidates = np.random.choice(len(population), size=tournament_size, replace=False)
        best_index = candidates[np.argmax([fitness[i] for i in candidates])]
        return population[best_index].copy()

    def _crossover(self, parent_a: np.ndarray,
                   parent_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(parent_a) < 2:
            return parent_a.copy(), parent_b.copy()
        point = np.random.randint(1, len(parent_a))
        child_a = np.concatenate([parent_a[:point], parent_b[point:]])
        child_b = np.concatenate([parent_b[:point], parent_a[point:]])
        return child_a.astype(np.int8), child_b.astype(np.int8)

    def _mutate(self, chromosome: np.ndarray, mutation_rate: float = 0.08) -> np.ndarray:
        mutated = chromosome.copy()
        for index in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[index] = 1 - mutated[index]
        if mutated.sum() == 0:
            mutated[np.random.randint(0, len(mutated))] = 1
        return mutated.astype(np.int8)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GeneticSVMPredictor":
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        if X.shape[1] == 0:
            raise ValueError("Feature matrix must contain at least one feature")
        if len(y) < 10:
            raise ValueError("Genetic algorithm requires at least 10 training rows")

        n_features = X.shape[1]
        cv = KFold(n_splits=min(self.cv_folds, len(y)), shuffle=True, random_state=self.random_seed)
        population = [self._make_chromosome(n_features) for _ in range(self.population_size)]
        fitness = [self._fitness(chromosome, X, y, cv) for chromosome in population]

        for generation in range(self.generations):
            next_population: List[np.ndarray] = []
            elite_index = int(np.argmax(fitness))
            next_population.append(population[elite_index].copy())

            while len(next_population) < self.population_size:
                parent_a = self._tournament(population, fitness)
                parent_b = self._tournament(population, fitness)
                child_a, child_b = self._crossover(parent_a, parent_b)
                child_a = self._mutate(child_a)
                child_b = self._mutate(child_b)
                next_population.extend([child_a, child_b])

            population = next_population[:self.population_size]
            fitness = [self._fitness(chromosome, X, y, cv) for chromosome in population]
            best_index = int(np.argmax(fitness))
            best_mask = self._selected_mask(population[best_index])
            best_score = float(np.sqrt(-fitness[best_index]))
            print("GA generation {}/{} - CV RMSE: {:.6f} - features: {}".format(
                generation + 1,
                self.generations,
                best_score,
                int(best_mask.sum())
            ))

        best_index = int(np.argmax(fitness))
        best_chromosome = population[best_index]
        best_mask = self._selected_mask(best_chromosome)
        self.selected_features = X.columns[best_mask].tolist()

        if not self.selected_features:
            self.selected_features = [X.columns[0]]

        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=self.C, epsilon=self.epsilon, gamma=self.gamma))
        ])
        self.model.fit(X[self.selected_features], y)
        self._details = {
            "encoding": "binary",
            "objective": "feature_selection",
            "selected_features": self.selected_features,
            "feature_count": len(self.selected_features),
            "selected_mask": best_mask.astype(int).tolist(),
            "C": self.C,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "cv_rmse": float(np.sqrt(-fitness[best_index]))
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        return self.model.predict(X[self.selected_features])

    def details(self) -> Dict[str, Any]:
        return self._details.copy()
