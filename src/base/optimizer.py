from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Dict

from src.base.types import T

# pylint: disable=too-few-public-methods
class Optimizer(ABC):
    """Abstract class for optimizers.
    Currently, you need to define the f and grad_f functions in the constructor.

    However `f` is not really used, consider removing in future.
    """

    def __init__(
        self,
        f: Callable,
        grad_f: Optional[Callable] = None,
        lr: float = 0.1,
        **kwargs: Dict[str, Any]
    ) -> None:
        self.f = f
        self.grad_f = grad_f
        self.lr = lr
        self.kwargs = kwargs

    @abstractmethod
    def step(
        self,
        weights: T,
        biases: Optional[T] = None,
    ) -> None:
        """Update the weights and biases.

        Args:
            weights (np.ndarray): The weights to update.
            biases (np.ndarray): The biases to update.

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError
