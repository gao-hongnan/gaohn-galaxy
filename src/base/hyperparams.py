"""Base Abstract Class for Hyperparameters.

Another way of handling this is using PyTorch Lightning's Mixin class,
which saves all the parameters passed to the class constructor as attributes
https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/core/mixins/hparams_mixin.py
def save_hyperparameters(self, ignore: Optional[List[Any]] = None):
    if ignore is None:
        ignore = []

    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    self.hparams = {
        k: v
        for k, v in local_vars.items()
        if k not in set(ignore + ["self"]) and not k.startswith("_")
    }
    for k, v in self.hparams.items():
        setattr(self, k, v)
"""
from __future__ import annotations

from typing import Any, Dict, Type

from pydantic import BaseModel  # pylint: disable=no-name-in-module


class Hyperparams(BaseModel):
    """Base Abstract Class for Hyperparameters."""

    random_state: int

    @classmethod
    def from_dict(cls: Type[Hyperparams], src: Dict[str, Any]) -> Hyperparams:
        """Creates a Pydantic instance from a dictionary."""
        return cls(**src)

    def to_dict(self) -> Dict[str, Any]:
        """Converts a Pydantic instance to a dictionary."""
        return self.dict()
