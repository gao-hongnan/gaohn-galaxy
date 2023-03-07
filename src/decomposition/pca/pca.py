from __future__ import annotations

import numpy as np
from src.base.estimator import BaseEstimator
from src.base.types import T
from src.utils.general import seed_all

def empirical_covariance_matrix(X: T) -> T:
    """Calculate the covariance matrix for the dataset X.
    X is assumed to be N x D where N is the number of samples."""
    n_samples = X.shape[0]
    covariance_matrix = (1 / (n_samples - 1)) * ((X - X.mean(axis=0)).T @ (X - X.mean(axis=0)))
    return np.array(covariance_matrix, dtype=float)


class PCA(BaseEstimator):
    """A method for doing dimensionality reduction by transforming the feature
    space to a lower dimensionality, removing correlation between features and
    maximizing the variance along each feature axis. This class is also used throughout
    the project to plot data.

    Principal component analysis (PCA) implementation.

    Transforms a dataset of possibly correlated values into n linearly
    uncorrelated components. The components are ordered such that the first
    has the largest possible variance and each following component as the
    largest possible variance given the previous components. This causes
    the early components to contain most of the variability in the dataset.

    Parameters
    ----------
    n_components : int
    solver : str, default 'svd'
        {'svd', 'eigen'}
    """

    def __init__(self, n_components: int, solver: str = "svd", random_state: int = 1992) -> None:
        self.n_components = n_components
        self.solver = solver
        self.random_state = random_state

        seed_all(self.random_state)

    def fit(self, X: T) -> PCA:
        """Fit the dataset to the number of principal components specified in the
        constructor and return the transformed dataset"""
        covariance_matrix = empirical_covariance_matrix(X)

        # Where (eigenvector[:,0] corresponds to eigenvalue[0])
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][: self.n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, : self.n_components]

        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)

        return X_transformed

    def transform(self, X, n_components):
        """Fit the dataset to the number of principal components specified in the
        constructor and return the transformed dataset"""
        covariance_matrix = empirical_covariance_matrix(X)

        # Where (eigenvector[:,0] corresponds to eigenvalue[0])
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]

        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)

        return X_transformed


if __name__ == "__main__":
    # TODO: populate these to notebook

    # X is a 9 x 2 matrix with 9 samples and 2 features x_1 and x_2
    X = np.array(
        [
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 6.0],
            [1.0, 1.0],
            [1.5, 1.6],
            [1.1, 0.9],
        ]
    )

    cov = empirical_covariance_matrix(X)
    print(f"Covariance matrix using empirical_covariance_matrix(): {cov}")

    cov_np = np.cov(X.T)
    print(f"Covariance matrix using numpy.cov(): {cov_np}")

    assert np.allclose(cov, cov_np)
