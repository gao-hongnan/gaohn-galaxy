"""Linear_R: y = wTx+b
Logistic_R: y = 1/(1+exp(-(wTx+b)))
Loss_Fn = Cross Entropy Loss - 1/N summation(y log y_pred + (1-y)log(1-y_pred))
Gradient Vector = (y-y_pred)X

DEBUG

epoch: 0 | loss: 232.70819390887343
epoch: 1000 | loss: 10.744077686087834
epoch: 2000 | loss: 8.993106726707099
epoch: 3000 | loss: 8.272247747702945
epoch: 4000 | loss: 7.86435449703676
my coef [[-0.50656544]
 [-0.34858508]
 [-0.58202777]
...

Accuracy score : 98.245614
Recall score : 98.734177
ROC score : 97.938517

[[34  1]
 [ 1 78]]
"""
from __future__ import annotations
from typing import Optional

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sklearn
from rich import print  # pylint: disable=redefined-builtin
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from src.optimizers.gd import GradientDescent
from src.activations.sigmoid import Sigmoid
from src.criterions.cross_entropy import BinaryCrossEntropy
from src.base.criterion import Criterion
from src.base.activation import Activation
from src.base.optimizer import Optimizer

from src.base.estimator import BaseEstimator
from src.base.types import T

# TODO:
# 1. Add support for multiclass classification
# 2. Explain why loss function and grad f need to take in w as args! This is because J(w)
#    is a function of w, and grad J(w) is the gradient of J(w) with respect to w.
# 3. Think of setting self.X = X, self.y = y as a new method? (e.g. self._set_X_y(X, y))
# 4. Check if you really need to average the gradient loss since our BCE already averaged.

# pylint: disable=too-many-instance-attributes, too-many-arguments
class LinearModel(BaseEstimator):
    def __init__(
        self,
        criterion: Criterion,
        activation: Activation,
        solver: str = "Batch Gradient Descent",
        has_intercept: bool = True,
        learning_rate: float = 1e-3,
        num_epochs: int = 1000,
    ):
        super().__init__()

        self.solver: str = solver
        self.num_epochs: int = num_epochs
        self.has_intercept: bool = has_intercept
        self.learning_rate: float = learning_rate

        self.coef_: Optional[T[float]] = None
        self.intercept_: Optional[float] = None
        self.optimal_betas: Optional[T[float]] = None
        self._fitted: bool = False

        self.optimizer: Optimizer

        self.criterion = criterion

        self.activation = activation

    def _loss(self, w):
        raise NotImplementedError

    def _grad_loss(self, w):
        raise NotImplementedError

    def _init_weights(self, X: T) -> T:
        """
        Initialize the weight and bias vector.

        Parameters:
                X (np.array): 2D numpy array (n_samples, n_features). Input Matrix of size m by n; where m is the number of samples, and n the number of features.

        Returns:
                uniform_weights (np.array): using He.Kaimin initialization

        """
        n_features = np.shape(X)[1]
        limit = 1 / np.sqrt(n_features)
        uniform_weights = np.random.uniform(-limit, limit, size=(n_features, 1))
        return uniform_weights

    def _check_and_reshape(self, X: T, y: T):
        """
        Check the shape of the inputs X & y; In particular, y must be reshaped to a row vector.

        1.  If X is 1D array, then it is simple logistic regression with 1 feature/variable,
            we need to make sure that X is reshaped to a 2D array/matrix.
            [1,2,3] (3,) -> [[1],[2],[3]] (3, 1) to fit the data.

        2.  If y is 1D array, which may usually be the case, we need to reshape it to 2D array.
            y = [1,0,1] (3,) -> y = [[1],[0],[1]] (3,1)


        Parameters:
            X (T): 2D numpy array (n_samples, n_features).
                Input Matrix of size N by D; where N is the number of samples, and D the number of features.

            y (T): 1D numpy array (n_samples, ).
                Input ground truth, also referred to as y of size N by 1.
        """
        if len(X.shape) == 1:
            X = np.reshape(X, newshape=(-1, 1))

        if len(y.shape) == 1:
            y = np.reshape(y, newshape=(1, -1))

        return X, y

    @staticmethod
    def _add_intercept(X: T) -> T:
        # X = np.insert(X, 0, 1, axis=1)
        # X = np.c_[np.ones(n_samples), X]
        b = np.ones([X.shape[0], 1])
        return np.concatenate([b, X], axis=1)

    def fit(self, X: T, y: Optional[T] = None):

        """
        Does not return anything. Instead it calculates the optimal beta coefficients for the Logistic Regression Model.
        The default solver will be Batch Gradient Descent where we optimize the weights by minimizing the cross-entropy loss function.

        Parameters:
                X (np.array): 2D numpy array (n_samples, n_features). Input Matrix of size m by n; where m is the number of samples, and n the number of features.

                y (np.array): 1D numpy array (n_samples,). Input ground truth, also referred to as y of size m by 1.

        Returns:
                self (MyLogisticRegression): Method for chaining

        Examples:
        --------
                >>> see main

        Explanation:
        -----------

        """

        X, y = self._check_and_reshape(X, y)

        if self.has_intercept:
            X = self._add_intercept(X)

        self.X = X  # MUST SET HERE AFTER INTERCEPT IS ADDED
        self.y = y

        n_samples, n_features = X.shape
        self.n_samples, self.n_features = n_samples, n_features
        # y must be a row vector with shape 1 x n_samples
        assert y.shape == (1, n_samples)

        self.optimal_betas = self._init_weights(X)
        # weight vector must be a column vector of shape (n_features, 1)
        assert self.optimal_betas.shape == (n_features, 1)

        for epoch in range(self.num_epochs):
            z = np.matmul(X, self.optimal_betas).T
            # z must be a row vector with shape 1 x n_samples
            assert z.shape == (1, n_samples)

            y_pred = self.activation(z)
            # y_pred must be a row vector with shape 1 x n_samples
            assert y_pred.shape == (1, n_samples)

            # TODO: consider using partial functions to pass in y and y_pred?
            f = self._loss
            grad_f = self._grad_loss

            GRADIENT_VECTOR = -np.matmul((y - y_pred), X).T
            # gradient vector must be a column vector of (n_features, 1)
            assert GRADIENT_VECTOR.shape == (n_features, 1)
            # we need to divide gradient vector by number of samples.
            # this is because each element inside the gradient vector is an accumulation/sum of across all samples.
            AVG_GRADIENT_VECTOR = (1 / n_samples) * GRADIENT_VECTOR

            if self.solver == "Batch Gradient Descent":
                self.optimizer = GradientDescent(
                    f, grad_f=grad_f, lr=self.learning_rate
                )

                # self.optimal_betas -= self.learning_rate * AVG_GRADIENT_VECTOR

                self.optimal_betas, _, _ = self.optimizer.step(self.optimal_betas)
                cross_entropy_loss = self.criterion(y, y_pred)

                if epoch % 1000 == 0:
                    print("epoch: {} | loss: {}".format(epoch, cross_entropy_loss))

            self.coef_ = self.optimal_betas[1:]
            self.intercept_ = self.optimal_betas[0]
            self._fitted = True

        return self

    def predict(self, X: T) -> T:
        """
        Predict using Logistic Regression - can only be called after fitting.
        Prediction formula will be using the sigmoid function for binary.

                Parameters:
                        X (np.array): 2D numpy array (n_samples, n_features). Input Matrix of size m by n; where m is the number of samples, and n the number of features.

                Returns:
                        y_logits (np.array): raw logits (probabilities).
                        y_pred (np.array): predictions in 0 and 1 for binary.
                -----------

        """
        if self.has_intercept:
            z = np.matmul(X, self.coef_) + self.intercept_  # z is a logit
            y_probs = self.activation(z)
        else:
            z = np.matmul(X, self.coef_)
            y_probs = self.activation(z)

        y_pred = np.where(y_probs < 0.5, 0, 1)

        return y_probs, y_pred


class BinaryLogisticRegression(LinearModel):
    def _loss(self, w):
        # 1. here w is both the weights and the bias for simplicity
        # 2. here _loss is \hat{\mathcal{J}}.
        # 3. why define again if we have cross_entropy_loss? This is because
        #    CE loss takes in y and y_pred as args, but in the context of
        #    ml, loss is a func of the weights w, not y and y_pred.

        loss = self.criterion(self.y, self.activation(np.matmul(self.X, w).T))
        return loss

    def _grad_loss(self, w):
        y_pred = self.activation(np.matmul(self.X, w).T)
        grad_loss = -np.matmul((self.y - y_pred), self.X).T
        mean_grad_loss = grad_loss / self.n_samples

        return mean_grad_loss


if __name__ == "__main__":

    """
    ================================
    Breast Cancer Classification Exercise
    ================================

    A tutorial exercise regarding the use of classification techniques on
    the Breast Cancer dataset.
    """
    np.random.seed(1930)
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=1930
    )
    print(y_train.shape)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    logreg = LogisticRegression(
        fit_intercept=True, random_state=1930, solver="sag", max_iter=1000, penalty=None
    )
    logreg.fit(X_train, y_train)
    print("sklearn coef", logreg.coef_, logreg.intercept_)
    print("SKLEARN Validation Accuracy: {}".format(logreg.score(X_val, y_val)))

    criterion = BinaryCrossEntropy()
    activation = Sigmoid()

    mylog = BinaryLogisticRegression(
        criterion=criterion,
        activation=activation,
        learning_rate=0.1,
        has_intercept=True,
        num_epochs=5000,
    )
    mylog.fit(X_train, y_train)

    print("my coef", mylog.coef_)
    logits, preds = mylog.predict(X_val)

    print(
        "\nAccuracy score : %f" % (sklearn.metrics.accuracy_score(y_val, preds) * 100)
    )
    print("Recall score : %f" % (sklearn.metrics.recall_score(y_val, preds) * 100))
    print("ROC score : %f\n" % (sklearn.metrics.roc_auc_score(y_val, preds) * 100))
    print(sklearn.metrics.confusion_matrix(y_val, preds))
    """
    ================================
    Digits Classification Exercise
    ================================

    A tutorial exercise regarding the use of classification techniques on
    the Digits dataset.

    This exercise is used in the :ref:`clf_tut` part of the
    :ref:`supervised_learning_tut` section of the
    :ref:`stat_learn_tut_index`.
    """
    # print(__doc__)

    # from sklearn import datasets, neighbors, linear_model

    # """
    # This below is a multiclass logistic regression using softmax. My code
    # not ready yet.
    # """

    # X_digits, y_digits = datasets.load_digits(return_X_y=True)

    # X_digits = X_digits / X_digits.max()

    # n_samples = len(X_digits)

    # X_train = X_digits[: int(0.9 * n_samples)]
    # y_train = y_digits[: int(0.9 * n_samples)]
    # X_test = X_digits[int(0.9 * n_samples) :]
    # y_test = y_digits[int(0.9 * n_samples) :]

    # knn = neighbors.KNeighborsClassifier()
    # logistic = linear_model.LogisticRegression(max_iter=1000)

    # print("KNN score: %f" % knn.fit(X_train, y_train).score(X_test, y_test))
    # print(
    #     "LogisticRegression score: %f"
    #     % logistic.fit(X_train, y_train).score(X_test, y_test)
    # )

    # hn_logreg = MyLogisticRegression(
    #     has_intercept=True,
    #     learning_rate=1e-5,
    #     solver="Gradient Descent",
    #     num_epochs=1000,
    # )
    # hn_logreg.fit(X_train, y_train)
    # ylogits, ypreds = hn_logreg.predict(X_test)
    # print(y_test)
    # print(hn_logreg.coef_)
    # print(ylogits)
    # print(sklearn.metrics.accuracy_score(y_test, ypreds))
