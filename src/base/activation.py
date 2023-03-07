from __future__ import annotations

from abc import ABC, abstractmethod

from src.base.types import T


class Activation(ABC):
    """
    Base class for activation functions.
    """

    @abstractmethod
    def __call__(self, z: T) -> T:
        """
        Compute the output of the activation function for a given input.

        Args:
            z: A numpy array of shape (batch_size, input_size).

        Returns:
            A numpy array of shape (batch_size, output_size).
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, z: T) -> T:
        """
        Compute the gradient of the activation function with respect to its input.

        Args:
            z: A numpy array of shape (batch_size, input_size).

        Returns:
            A numpy array of shape (batch_size, input_size).
        """
        raise NotImplementedError
