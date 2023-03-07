import numpy as np

from src.base.activation import Activation
from src.base.types import T


class Softmax(Activation):
    """
    Softmax activation function.
    """

    def __call__(self, z: T) -> T:
        """
        Compute the softmax function for a given input.

        Args:
            z: A numpy array of shape (batch_size, input_size).

        Returns:
            A numpy array of shape (batch_size, output_size).
        """
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        g = numerator / denominator
        return g

    def gradient(self, z: T) -> T:
        """
        Compute the derivative of the softmax function with respect to its input.

        Args:
            x: A numpy array of shape (batch_size, input_size).

        Returns:
            A numpy array of shape (batch_size, input_size).
        """
        g = self.__call__(z)
        dg_dz = g * (1 - g)
        return dg_dz
