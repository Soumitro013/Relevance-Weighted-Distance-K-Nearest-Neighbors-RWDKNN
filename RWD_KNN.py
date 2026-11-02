
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


class RWDKNN:
    """
    Relevance-Weighted Distance KNN Regression.

    1. Standardizes predictors.
    2. Calculates Spearman-based feature weights.
    3. Uses weighted Euclidean distance.
    4. Applies inverse-distance-weighted prediction.
    5. Selects K using chronological cross-validation.
    """

    def __init__(
        self,
        k_values=None,
        epsilon=1e-8
    ):

        self.k_values = (
            k_values
            if k_values is not None
            else [1, 3, 5, 7, 9, 11]
        )

        self.epsilon = epsilon

        self.scaler = None
        self.feature_weights = None
        self.X_train = None
        self.y_train = None
        self.best_k = None
        self.feature_names = None


    # Calculate feature relevance weights


    def _calculate_weights(self, X, y):

        p = X.shape[1]

        relevance = np.zeros(p)

        for j in range(p):

            # Spearman correlation
            result = spearmanr(
                X[:, j],
                y,
                nan_policy="omit"
            )

            correlation = result.statistic

            if np.isfinite(correlation):
                relevance[j] = abs(correlation)

        # Normalize relevance values
        weights = relevance + self.epsilon

        weights = weights / np.sum(weights)

        return weights


    # Fit model


    def fit(self, X, y):

        if isinstance(X, pd.DataFrame):

            self.feature_names = list(X.columns)
            X = X.to_numpy()

        else:

            self.feature_names = [
                f"X{i}" for i in range(X.shape[1])
            ]

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError(
                "X must be 2D and y must be 1D."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must have equal lengths."
            )

        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(
                "Input data contains missing or infinite values."
            )

        # Standardization
        self.scaler = StandardScaler()

        X_scaled = self.scaler.fit_transform(X)

        # Feature relevance weights
        self.feature_weights = (
            self._calculate_weights(X_scaled, y)
        )

        # Store training observations
        self.X_train = X_scaled
        self.y_train = y

        return self


    # Weighted Euclidean distance


    def _weighted_distance(self, x):

        differences = (
            self.X_train - x
        )

        distances = np.sqrt(
            np.sum(
                self.feature_weights
                * differences ** 2,
                axis=1
            )
        )

        return distances


    # Prediction for one observation


    def predict_one(self, x, k=None):

        if self.X_train is None:
            raise RuntimeError(
                "Fit the model before prediction."
            )

        if k is None:
            k = self.best_k

        if k is None:
            k = 5

        if k < 1 or k > len(self.X_train):
            raise ValueError(
                "Invalid number of neighbors."
            )

        x = np.asarray(
            x,
            dtype=float
        ).reshape(1, -1)

        x_scaled = self.scaler.transform(x)[0]

        distances = self._weighted_distance(
            x_scaled
        )

        # Select nearest neighbors
        neighbor_indices = np.argsort(
            distances
        )[:k]

        neighbor_distances = (
            distances[neighbor_indices]
        )

        neighbor_targets = (
            self.y_train[neighbor_indices]
        )

        # Exact-match handling
        exact_matches = (
            neighbor_distances <= self.epsilon
        )

        if np.any(exact_matches):

            return float(
                np.mean(
                    neighbor_targets[exact_matches]
                )
            )

        # Inverse-distance weighting
        weights = (
            1.0 /
            (neighbor_distances + self.epsilon)
        )

        prediction = np.sum(
            weights * neighbor_targets
        ) / np.sum(weights)

        return float(prediction)


    # Batch prediction


    def predict(self, X, k=None):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        X = np.asarray(
            X,
            dtype=float
        )

        return np.array([
            self.predict_one(row, k=k)
            for row in X
        ])


    # Automatic selection of K


    def select_k(
        self,
        X,
        y,
        n_splits=5
    ):

        X = np.asarray(
            X,
            dtype=float
        )

        y = np.asarray(
            y,
            dtype=float
        )

        splitter = TimeSeriesSplit(
            n_splits=n_splits
        )

        cv_results = []

        for k in self.k_values:

            fold_errors = []

            for train_idx, val_idx in splitter.split(X):

                X_train = X[train_idx]
                y_train = y[train_idx]

                X_val = X[val_idx]
                y_val = y[val_idx]

                # Skip folds with insufficient
                # training observations.
                if k > len(train_idx):
                    continue

                model = RWDKNN(
                    k_values=self.k_values,
                    epsilon=self.epsilon
                )

                model.fit(
                    X_train,
                    y_train
                )

                predictions = model.predict(
                    X_val,
                    k=k
                )

                error = mean_squared_error(
                    y_val,
                    predictions
                )

                fold_errors.append(error)

            if fold_errors:

                cv_results.append({
                    "K": k,
                    "CV_MSE": np.mean(fold_errors)
                })

        if not cv_results:
            raise ValueError(
                "No valid K values. Reduce K or "
                "provide more training observations."
            )

        results = pd.DataFrame(
            cv_results
        )

        best_index = results["CV_MSE"].idxmin()

        self.best_k = int(
            results.loc[best_index, "K"]
        )

        return results
