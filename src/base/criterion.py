from __future__ import annotations

from abc import ABC, abstractmethod

from src.base.types import T


class Criterion(ABC):
    @abstractmethod
    def __call__(self, y_trues: T, y_preds: T) -> float:
        """Computes Loss."""
