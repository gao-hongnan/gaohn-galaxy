import random
import statistics
from functools import partial
from typing import Union

import numpy as np

from src.base.estimator import BaseEstimator
from src.base.types import T
from src.metrics.pairwise.distance import euclidean_distance, manhattan_distance

# TODO: to be consistent with my other code (K-Means)


class KNNBase(BaseEstimator):
    """KNN Algorithm from Scratch.

    Parameters
    ----------
    num_neighbours : int
        K in K-Nearest-Neighbours.

    metric : str, optional, default: 'euclidean_distance'
        The metric used to calculate the distance between each sample.

    algorithm : str, optional, default: 'brute'
        The algorithm used to find the nearest neighbours.
    """

    def __init__(
        self, num_neighbours: int, metric: str = "euclidean", algorithm: str = "brute"
    ):
        self._K = num_neighbours
        self.metric = metric
        self.distance = self._get_distance_metric()  # get distance fn based on metric

        self.algorithm = algorithm  # current only support brute force

    def aggregate(self, neighbor_labels: T):
        """Aggregate the top K nearest neighbours, if classification, return the most
        common class, if regression, return the mean."""
        raise NotImplementedError

    def _get_distance_metric(self) -> Union[euclidean_distance, manhattan_distance]:
        if self.metric == "euclidean":
            return partial(euclidean_distance, squared=False)
        if self.metric == "manhattan":
            return manhattan_distance
        raise ValueError(f"{self.metric} is not supported.")

    def check_shape(self, X: np.ndarray = None, y: np.ndarray = None) -> None:
        """Always call `np.asarray()` on the incoming inputs as this is a good way to check whether the user input funny data types. And it also conveniently turns list into nparray.

        Args:
            X (np.ndarray, optional): [description]. Defaults to None.
            y (np.ndarray, optional): [description]. Defaults to None.
        """
        if X is not None:
            X = np.asarray(X)
            assert len(X.shape) == 2, "The input X matrix should be a 2-d array!"

        if y is not None:
            assert (
                len(y.shape) == 1
            ), "Both the y_train and y_pred array should be a 1-d array!"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # TODO: check what sklearn in fit.
        self.check_shape(X=X_train, y=y_train)

    def predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        """The following steps details one cycle of classification. The regression follows just by changing the mode of the init.

        line 1-2: Check shape of input arrays.

        line 3: Initialize an empty y_preds array to be same size of X_test. This is a 1d-array of predictions in classes (int) which corresponds to X_test.

        line 4 - end: We explain just the first loop, and then the subsequent loops is the same.

            1. [self.distance(test_sample, x) for x in X_train]: take the first unclassified sample x_q in X_test, this code computes the distance of x_q with ALL of X_train data and store in a list.
               Example: test = [1,2,3], train = [[1,1,1], [2,2,2]] we then compute distance of test with each x_i of X_train. Final output is a 1d array [0.1, 0.3, ...]

            2. np.argsort([self.distance(test_sample, x) for x in X_train]): applied np.argsort on point 1 where we sort the list in 1. This array returns the index of the sorted array. This is important to know.
               Example: x = np.array([30, 10, 20]) -> np.argsort(x) -> array([1, 2, 0])
               Note this is sorted according to index because argsort first sort it according to ascending order, which should be [10, 20, 30].
               However, the function returns [1, 2, 0] because they know the smallest number is 10, and it resides at index 1, second number is 20, residing at index 2, etc.

            3. np.argsort([self.distance(test_sample, x) for x in X_train])[:self._K]: Simple slicing on point 2 to get the top k nearest neighbours' index.

            4. k_nearest_neighbors_classes = np.array([y_train[i] for i in k_nearest_neighbors_idx]): This simply finds the corresponding true labels corresponding to the indexes found in 3.

            5. y_preds[i] = self._vote(k_nearest_neighbors_classes): Now, call _vote on this array, simply put, if this array is [1,1,2,3,1,1,3], then by majority vote, class 1 is the chosen prediction for this test sample x_q.


        Args:
            X_train (np.ndarray): [description]
            y_train (np.ndarray): [description]
            X_test (np.ndarray): [description]

        Returns:
            np.ndarray: [description]
        """
        self.check_shape(X=X_test, y=None)

        y_preds = np.empty(X_test.shape[0])

        for i, test_sample in enumerate(X_test):

            k_nearest_neighbors_idx = np.argsort(
                [self.distance(test_sample, x) for x in X_train]
            )[: self._K]

            k_nearest_neighbors_classes = np.array(
                [y_train[i] for i in k_nearest_neighbors_idx]
            )
            y_preds[i] = self.aggregate(k_nearest_neighbors_classes)
        return y_preds

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return probability estimates for the test data X.

        Only Applicable for Classification

        Args:
            X_test (np.ndarray): [description]

        Returns:
            p: np.ndarray: [description]
        """


class KNNClassifier(KNNBase):
    def _vote(self, neighbor_labels):
        """Return the most common class among the neighbor samples"""
        frequency_counts = np.bincount(neighbor_labels.astype("int"))
        majority_vote = np.argmax(frequency_counts)
        return majority_vote

    def aggregate(self, neighbor_labels: T):
        return self._vote(neighbor_labels)


class KNNRegressor(KNNBase):
    def _mean(self, neighbor_labels):
        """Average the top K nearest neighbours in Regression"""
        return sum(neighbor_labels) / len(neighbor_labels)

    def aggregate(self, neighbor_labels: T):
        return self._mean(neighbor_labels)


def partition(arr, lo, high):
    rand = random.randint(lo, high)
    arr[rand], arr[high] = arr[high], arr[rand]
    pivot = lo
    for i in range(lo, high):
        if arr[i][0] < arr[high][0]:
            arr[i], arr[pivot] = arr[pivot], arr[i]
            pivot += 1
    arr[pivot], arr[high] = arr[high], arr[pivot]
    return pivot


def quickselect(arr, lo, hi, k):
    while True:
        pivot = partition(arr, lo, hi)
        if pivot < k:
            lo = pivot + 1
        elif pivot > k:
            hi = pivot - 1
        else:
            return


def KNN_quickselect(X, y, x_test, k):
    # first, calculate the distance from the new point
    # to all other points in the dataset (x_test)
    distances = []  # stores (dist, class)
    for i in range(len(X)):
        d = euclidean_distance(X[i], x_test)
        distances.append((d, y[i]))

    # second, sort and then store the K nearest neighbors
    quickselect(distances, 0, len(distances), k)
    # Get the most popular class
    classes = [c for dist, c in distances[:k]]
    y_pred = statistics.mode(classes)
    return y_pred
