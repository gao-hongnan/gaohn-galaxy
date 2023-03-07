import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

from autograd import grad

from src.base.optimizer import Optimizer
from src.base.types import T


# pylint: disable=too-few-public-methods
class GradientDescent(Optimizer):
    """Gradient descent optimizer."""

    weights_history: List[T]

    def __init__(
        self,
        f: Callable,
        grad_f: Optional[Callable] = None,
        lr: float = 0.1,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(f, grad_f, lr, **kwargs)
        self.weights_history = []

        # FIXME: not working for some.
        if grad_f is None:
            print("grad_f is None, using autograd")
            self.grad_f = grad(f)  # pylint: disable=no-value-for-parameter

    def step(
        self,
        weights: T,
        biases: Optional[T] = None,
    ) -> Tuple[T, T, List[T]]:
        # grad vec at each point
        gradient_vector = self.grad_f(weights)
        weights -= self.lr * gradient_vector

        if biases is not None:
            raise NotImplementedError("Biases are not implemented yet.")

        # print(f"weights {weights}, biases {biases}")

        # must use copy if not weights (np array) mutatable:
        # https://tinyurl.com/bdff69by
        self.weights_history.append(copy.copy(weights))
        return weights, biases, self.weights_history
