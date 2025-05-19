__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images
import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels, distance_metric='euclidean', normalize=False):
        """
        Initialize KNN classifier.

        Args:
            train_data: Training data (PxMxNx3 matrix for color images)
            labels: Array of labels for training data
            distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine', 'chebyshev')
            normalize: Whether to normalize the data (default: False)
        """
        self._init_train(train_data, normalize)
        self.labels = np.array(labels)
        self.valid_metrics = ['euclidean', 'manhattan', 'cosine', 'chebyshev']
        if distance_metric not in self.valid_metrics:
            raise ValueError(f"Invalid distance metric. Choose from {self.valid_metrics}")
        self.distance_metric = distance_metric
        self.normalize = normalize

    def _init_train(self, train_data, normalize):
        """
        Initialize and preprocess training data.

        Args:
            train_data: Training data (PxMxNx3 matrix for color images)
            normalize: Whether to normalize the data
        """
        train_data = np.array(train_data, dtype=np.float32)

        # Convert RGB to grayscale if needed
        if train_data.ndim == 4 and train_data.shape[-1] == 3:
            train_data = np.mean(train_data, axis=-1)

        # Flatten images
        P, M, N = train_data.shape
        self.train_data = train_data.reshape(P, M * N)

        # Store original shape for test data reshaping
        self.original_shape = (M, N)

        # Normalize if requested
        if normalize:
            self.mean = np.mean(self.train_data, axis=0)
            self.std = np.std(self.train_data, axis=0)
            self.train_data = (self.train_data - self.mean) / (self.std + 1e-8)  # Add epsilon to avoid division by zero

    def get_k_neighbours(self, test_data, k):
        """
        Find k nearest neighbors for test data.

        Args:
            test_data: Test data (NxMxNx3 matrix for color images)
            k: Number of neighbors to find

        Sets:
            self.neighbors: Labels of k nearest neighbors (NxK matrix)
            self.distances: Distances to k nearest neighbors (NxK matrix)
        """
        test_data = np.array(test_data, dtype=np.float32)

        # Convert RGB to grayscale if needed
        if test_data.ndim == 4 and test_data.shape[-1] == 3:
            test_data = np.mean(test_data, axis=-1)

        # Flatten test data
        N, M, P = test_data.shape
        test_data_flat = test_data.reshape(N, M * P)

        # Normalize test data if training data was normalized
        if self.normalize:
            test_data_flat = (test_data_flat - self.mean) / (self.std + 1e-8)

        # Calculate distances
        metric_map = {
            'manhattan': 'cityblock',
            'euclidean': 'euclidean',
            'cosine': 'cosine',
            'chebyshev': 'chebyshev'
        }
        distances = cdist(test_data_flat, self.train_data, metric=metric_map[self.distance_metric])

        # Get k nearest neighbors
        nearest_indices = np.argsort(distances, axis=1)[:, :k]
        self.neighbors = self.labels[nearest_indices]
        self.distances = np.take_along_axis(distances, nearest_indices, axis=1)

    def get_class(self, weights='uniform'):
        """
        Get predicted class using majority voting.

        Args:
            weights: 'uniform' or 'distance' for weighted voting

        Returns:
            Predicted classes for test data
        """
        if weights == 'distance' and hasattr(self, 'distances'):
            # Weighted voting using inverse distances
            weights = 1 / (self.distances + 1e-8)
            predicted_classes = []

            for i, (neighbor_row, weight_row) in enumerate(zip(self.neighbors, weights)):
                unique_labels = np.unique(neighbor_row)
                label_scores = []

                for label in unique_labels:
                    mask = (neighbor_row == label)
                    score = np.sum(weight_row[mask])
                    label_scores.append((label, score))

                # Get label with highest score
                predicted_classes.append(max(label_scores, key=lambda x: x[1])[0])

            return np.array(predicted_classes)
        else:
            # Standard majority voting
            predicted_classes = []
            for row in self.neighbors:
                label_counts = {}
                for label in row:
                    label_counts[label] = label_counts.get(label, 0) + 1
                majority_label = max(label_counts.items(), key=operator.itemgetter(1))[0]
                predicted_classes.append(majority_label)
            return np.array(predicted_classes)

    def predict(self, test_data, k, weights='uniform'):
        """
        Predict classes for test data.

        Args:
            test_data: Test data (NxMxNx3 matrix for color images)
            k: Number of neighbors to consider
            weights: 'uniform' or 'distance' for weighted voting

        Returns:
            Predicted classes for test data
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class(weights)

    @staticmethod
    def cross_val_score(X, y, k_values, n_splits=5, metric='accuracy', distance_metric='euclidean', normalize=False):
        """
        Perform k-fold cross validation to evaluate KNN performance.

        Args:
            X: Input data
            y: Labels
            k_values: List of k values to evaluate
            n_splits: Number of folds
            metric: Evaluation metric ('accuracy')
            distance_metric: Distance metric to use
            normalize: Whether to normalize data

        Returns:
            Dictionary of {k: average_score}
        """
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits)

        results = {k: [] for k in k_values}

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for k in k_values:
                knn = KNN(X_train, y_train, distance_metric=distance_metric, normalize=normalize)
                pred = knn.predict(X_test, k)
                accuracy = np.mean(pred == y_test)
                results[k].append(accuracy)

        return {k: np.mean(scores) for k, scores in results.items()}

    def evaluate(self, test_data, test_labels, k, weights='uniform'):
        """
        Evaluate model performance on test data.

        Args:
            test_data: Test data
            test_labels: True labels for test data
            k: Number of neighbors
            weights: 'uniform' or 'distance' for weighted voting

        Returns:
            accuracy: Classification accuracy
        """
        predictions = self.predict(test_data, k, weights)
        return np.mean(predictions == test_labels)