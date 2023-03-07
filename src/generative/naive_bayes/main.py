"""Main script to run the Naive Bayes algorithm."""
from __future__ import annotations

import warnings
import numpy as np

from sklearn.datasets import load_iris, make_blobs
from sklearn.naive_bayes import GaussianNB
from rich import print  # pylint: disable=redefined-builtin

from src.generative.naive_bayes.naive_bayes import NaiveBayesGaussian
from src.utils.general import (
    plot_classifier_decision_boundary,
    run_classifier,
    seed_all,
)

warnings.filterwarnings("ignore", category=UserWarning)


def run_naive_bayes_sanity_check(
    my_estimator: NaiveBayesGaussian, sklearn_estimator: GaussianNB
) -> None:
    """Run sanity check on our Naive Bayes implementation with sklearn's implementation by
    comparing the **fitted** parameters of the two estimators.

    Therefore, both estimators should be fitted before calling this function.

    NOTE:
        Recall our theta is of shape (num_classes, num_features, 2)
        where 3x4x2 is (3 classes, 4 features, 2 parameters)
        so first class is 4x2 and the column is the mean of each feature for class 1
        and the second column is the variance of each feature for class 1
        coincides with sklearn so we gucci.
    """

    # assert my_estimator.is_fitted, "My estimator is not fitted."

    my_pi = my_estimator.pi
    my_theta = my_estimator.theta
    my_mean = my_theta[:, :, 0]
    my_covariance = my_theta[:, :, 1]
    print(f"My Pi:\n{my_pi}")
    print(f"My Mean:\n{my_mean}")
    print(f"My Covariances:\n{my_covariance}")

    sk_pi = sklearn_estimator.class_prior_
    sk_theta = sklearn_estimator.theta_
    sk_covariance = sklearn_estimator.var_

    print(f"Sklearn Pi:\n{sk_pi}")
    print(f"Sklearn Mean:\n{sk_theta}")
    print(f"Sklearn Covariances:\n{sk_covariance}")

    np.testing.assert_allclose(my_pi, sk_pi)
    np.testing.assert_allclose(my_mean, sk_theta)
    np.testing.assert_allclose(my_covariance, sk_covariance)


if __name__ == "__main__":
    seed_all(1992)
    X, y = load_iris(return_X_y=True)
    class_names = ["setosa", "versicolor", "virginica"]

    gnb = NaiveBayesGaussian(random_state=1992, num_classes=3)
    gnb = run_classifier(
        gnb,
        X,
        y,
        test_size=0.2,
        random_state=0,
        class_names=class_names,
    )

    sk_gnb = GaussianNB(var_smoothing=0)
    sk_gnb = run_classifier(
        sk_gnb,
        X,
        y,
        test_size=0.2,
        random_state=0,
        class_names=class_names,
    )

    run_naive_bayes_sanity_check(gnb, sk_gnb)

    # plotting IRIS decision boundary requires us
    # to only subset the data to two features.

    # plot the decision boundary for IRIS with 2 features
    X_2d = X[:, :2]
    plot_classifier_decision_boundary(gnb, X_2d, y)
    plot_classifier_decision_boundary(sk_gnb, X_2d, y)

    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_blobs(
        n_samples=500, centers=2, n_features=2, random_state=1992, return_centers=False
    )

    gnb = NaiveBayesGaussian(random_state=1992, num_classes=2)
    plot_classifier_decision_boundary(gnb, X, y)
