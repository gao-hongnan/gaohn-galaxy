"""Base Abstract Class for Estimators.

Design Pattern: Template/Strategy/Learner Pattern
For a more sophisicated design, refer to scikit-learn's OOP paradigm
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py

References:
    - https://towardsdatascience.com/how-the-strategy-design-pattern-can-help-you-quickly-evaluate-alternative-models-66e0f625016f
    - Scikit-learn's OOP paradigm
"""
from __future__ import annotations

from typing import Optional

from abc import ABC, abstractmethod

from src.base.types import T


class BaseEstimator(ABC):
    """Base Abstract Class for Estimators.

    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    Parameters
    ----------
    hyperparameters : Hyperparams, optional (default=None)
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    """

    def __repr__(self) -> str:
        """Return the string representation of the estimator.

        Returns
        -------
        repr : str
            The string representation of the estimator.
        """
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Return the string representation of the estimator.

        Returns
        -------
        str : str
            The string representation of the estimator.
        """
        return self.__repr__()

    def __eq__(self, other: BaseEstimator) -> bool:
        """Check if two estimators are equal.

        Parameters
        ----------
        other : BaseEstimator
            The other estimator.

        Returns
        -------
        eq : bool
            True if the estimators are equal, False otherwise.
        """
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        """Get the hash of the estimator.

        Returns
        -------
        hash : int
            The hash of the estimator.
        """
        return hash(tuple(sorted(self.__dict__.items())))

    @abstractmethod
    def fit(self, X: T, y: Optional[T] = None) -> BaseEstimator:
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        self : object
            Returns self.
        """

    @abstractmethod
    def predict(self, X: T) -> T:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Predicted class label per sample.
        """

    def score(self, X: T, y: T) -> float:
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """


class TestBaseEstimator(BaseEstimator):
    def fit(self, X: T, y: T = None) -> BaseEstimator:
        return self

    def predict(self, X: T) -> T:
        return X

    def score(self, X: T, y: T) -> float:
        return 1.0
