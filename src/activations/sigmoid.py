import numpy as np

from src.base.activation import Activation
from src.base.types import T


class Sigmoid(Activation):
    """
    Sigmoid activation function.

    Computes the gradient of the sigmoid function with respect to the input z.
    This is useful since in the backward pass for say
    Logistic Regression's Cross-Entropy Loss,
    dl/dz is needed in the chain rule, and dl/dz = dl/dA * dA/dz
    where A is y_pred is sigmoid(z).
    Consequently, dA/dz makes use of the gradient of the sigmoid function.
    """

    def __call__(self, z: T) -> T:
        """
        Compute the sigmoid function for a given input.

        Args:
            z: A numpy array of shape (batch_size, input_size).

        Returns:
            A numpy array of shape (batch_size, output_size).
        """
        g = 1 / (1 + np.exp(-z))
        return g

    def gradient(self, z: T) -> T:
        """
        Compute the derivative of the sigmoid function with respect to its input.

        Args:
            z: A numpy array of shape (batch_size, input_size).

        Returns:
            A numpy array of shape (batch_size, input_size).
        """
        g = self.__call__(z)
        dg_dz = g * (1 - g)
        return dg_dz
