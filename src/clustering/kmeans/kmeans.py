"""K-Means Implementation.

References:
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- https://github.com/rushter/MLAlgorithms/blob/master/mla/kmeans.py
"""
from __future__ import annotations

from functools import partial
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from src.base.estimator import BaseEstimator
from src.base.types import T
from src.metrics.pairwise.distance import euclidean_distance, manhattan_distance
from src.utils.general import seed_all


# pylint: disable=invalid-name,too-many-instance-attributes,too-many-arguments
class KMeansLloyd(BaseEstimator):
    """K-Means Lloyd's algorithm.

    Parameters
    ----------
    num_clusters : int, optional, default: 3
        The number of clusters to form as well as the number of
        centroids to generate.

    init : str, optional, default: "random"
        Method for initialization, either 'random' or 'k-means++'.

    max_iter : int, optional, default: 100
        Maximum number of iterations of the k-means algorithm for
        a single run.

    metric : str, optional, default: "euclidean"
        The distance metric used to calculate the distance between each
        sample and the centroid.

    tol : float, optional, default: 1e-8
        The tolerance with regards to the change in the within-cluster
        sum-squared-error to declare convergence.

    random_state : int, optional, default: 42
        Seed for the random number generator used during the
        initialization of the centroids.
    """

    def __init__(
        self,
        num_clusters: int = 3,
        init: str = "random",
        max_iter: int = 100,
        metric: str = "euclidean",
        tol: float = 1e-8,
        random_state: int = 42,
    ) -> None:

        self._K = num_clusters  # K
        self.init = init  # random init
        self.max_iter = max_iter

        self.metric = metric
        self.distance = self._get_distance_metric()  # get distance fn based on metric

        self.tol = tol

        self.random_state = random_state
        seed_all(self.random_state)

        self._reset_clusters()  # initializes self._C = {C_1=[], C_2=[], ..., C_k=[]}

        self._N: int
        self._D: int
        self.t: int  # iteration counter
        self._labels: T  # N labels np.zeros(shape=(self._N))
        self._centroids: T  # np.zeros(shape=(self._K, self.num_features)) KxD matrix
        self._inertia: T
        self._inertias: T  # np.zeros(shape=(self._N)) N inertias

    @property
    def num_clusters(self) -> int:
        """Property to get the number of clusters K."""
        return self._K

    @property
    def num_features(self) -> int:
        """Property to get the number of features D."""
        return self._D

    @property
    def num_samples(self) -> int:
        """Property to get the number of samples N."""
        return self._N

    @property
    def clusters(self) -> T:
        """Property to get the clusters, this is our C."""
        return self._C

    @property
    def labels(self) -> T:
        """Property to get the labels of the samples."""
        return self._labels.astype(int)

    @property
    def centroids(self) -> T:
        """Property to get the centroids."""
        return self._centroids

    @property
    def inertia(self) -> T:
        """Property to get the inertia."""
        return self._inertia

    def _reset_clusters(self) -> None:
        """Reset clusters."""
        self._C = {k: [] for k in range(self._K)}

    def _reset_inertias(self) -> None:
        """Reset inertias.

        This initializes self._inertias, and has shape N as it is the outer summation
        of our cost function J. We sum over all samples N to get self._inertia.
        """
        self._inertias = np.zeros(self._N)  # reset mechanism so don't accumulate

    def _reset_labels(self) -> None:
        """Reset labels."""
        self._labels = np.zeros(self._N)  # reset mechanism so don't accumulate

    def _init_centroids(self, X: T) -> None:
        self._centroids = np.zeros(shape=(self._K, self._D))  # KxD matrix
        if self.init == "random":
            for k in range(self._K):
                self._centroids[k] = X[np.random.choice(range(self._N))]
        else:
            raise ValueError(f"{self.init} is not supported.")

    def _get_distance_metric(self) -> Union[euclidean_distance, manhattan_distance]:
        if self.metric == "euclidean":
            return partial(euclidean_distance, squared=False)
        if self.metric == "manhattan":
            return manhattan_distance
        raise ValueError(f"{self.metric} is not supported.")

    def _compute_argmin_assignment(self, x: T, centroids: T) -> Tuple[int, float]:
        """Compute the argmin assignment for a single sample x.

        In other words, for a single sample x, compute the distance between x and
        each centroid mu_1, mu_2, ..., mu_k. Then, return the index of the centroid
        that is closest to x, and the distance between x and that centroid.

        This step has O(K) complexity, where K is the number of clusters. However,
        to be more pedantic, the self.distance also has O(D) complexity,
        where D is the number of features, so it can be argued that this step has
        O(KD) complexity.
        """
        min_index = None
        min_distance = np.inf
        for k, centroid in enumerate(centroids):
            distance = self.distance(x, centroid)
            if distance < min_distance:
                # now min_distance is the best distance between x and mu_k
                min_index = k
                min_distance = distance
        return min_index, min_distance

    def _assign_samples(self, X: T, centroids: T) -> None:
        """Assignment step: assigns samples to clusters.

        This step has O(NK) complexity since we are looping over
        N samples, and for each sample, _compute_argmin_assignment requires
        O(K) complexity."""
        self._reset_inertias()  # reset the inertias to [0, 0, ..., 0]
        self._reset_labels()  # reset the labels to [0, 0, ..., 0]
        for sample_index, x in enumerate(X):
            min_index, min_distance = self._compute_argmin_assignment(x, centroids)
            # fmt: off
            self._C[min_index].append(x) # here means append the data point x to the cluster C_k
            self._inertias[sample_index] = min_distance  # the cost of the sample x
            self._labels[sample_index] = int(min_index)  # the label of the sample x
            # fmt: on

    def _update_centroids(self, centroids: T) -> T:
        """Update step: update the centroid with new cluster mean."""
        for k, cluster in self._C.items():  # for k and the corresponding cluster C_k
            mu_k = np.mean(cluster, axis=0)  # compute the mean of the cluster
            centroids[k] = mu_k  # update the centroid mu_k
        return centroids

    def _has_converged(self, old_centroids: T, updated_centroids: T) -> bool:
        """Checks for convergence.
        If the centroids are the same, then the algorithm has converged.
        You can also check convergence by comparing the SSE."""
        return np.allclose(updated_centroids, old_centroids, atol=self.tol)

    def fit(self, X: T) -> None:  # FIXME: violated Liskov Substitution Principle
        """Fits K-Means to the data, after which the model will have
        centroids and labels."""
        # fmt: off
        self._N, self._D = X.shape  # N=num_samples, D=num_features

        # step 1. Initialize self._centroids of shape KxD.
        self._init_centroids(X)

        # enter iteration of step 2-3 until convergence
        for t in range(self.max_iter):
            # copy the centroids and denote it as old_centroids
            # note that in t+1 iteration, we will update the old_centroids
            # to be the self._centroid in t iteration
            old_centroids = self._centroids.copy()

            # step 2: assignment step
            self._assign_samples(X, old_centroids)

            # step 3: update step (careful of mutation without copy because I am doing centroids[k] = mu_k)
            # updated_centroids = self._update_centroids(old_centroids.copy())
            self._centroids = self._update_centroids(old_centroids.copy())

            # step 4: check convergence
            if self._has_converged(old_centroids, self._centroids):
                print(f"Converged at iteration {t}")
                self.t = t  # assign iteration index, used for logging
                break

            # do not put self._centroids here because if self._has_converged is True,
            # then this step is not updated, causing the final centroids to be the old_centroids
            # instead of the updated_centroids, especially problematic if tolerance>0
            # self._centroids = updated_centroids  # assign updated centroids to self._centroids if not converged
            self._reset_clusters()  # reset the clusters if not converged

        # step 5: compute the final cost on the converged centroids
        self._inertia = np.sum(self._inertias)
        # fmt: on

    def predict(self, X: T) -> T:
        """Predict cluster labels for samples in X where X can be new data or training data."""
        y_preds = np.array(
            [self._compute_argmin_assignment(x, self._centroids)[0] for x in X]
        )
        return y_preds


def elbow_method(X: T, min_clusters: int = 1, max_clusters: int = 10) -> None:
    """Elbow method to find the optimal number of clusters.
    The optimal number of clusters is where the elbow occurs.
    The elbow is where the SSE starts to decrease in a linear fashion."""
    inertias = []
    for k in range(min_clusters, max_clusters + 1):
        print(f"Running K-Means with {k} clusters")
        kmeans = KMeansLloyd(
            num_clusters=k, init="random", max_iter=500, random_state=1992
        )
        kmeans.fit(X)
        inertias.append(kmeans.inertia)
    plt.plot(range(min_clusters, max_clusters + 1), inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("Intertia/Distortion/SSE")
    plt.tight_layout()
    plt.show()


# pylint: disable=unused-argument
def kmeans_vectorized(X: T, num_clusters: int, max_iter: int = 500) -> T:
    """Implement K-Means using vectorized operations."""
