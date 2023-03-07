"""Type hints."""
from __future__ import annotations

from typing import TypeVar

import numpy as np
import torch

# indicates that the input type and the output type are the same
# i.e. predict(self, X: T) -> T means torch.Tensor -> torch.Tensor
T = TypeVar("T", np.ndarray, torch.Tensor)
