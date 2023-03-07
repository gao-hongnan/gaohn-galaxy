"""Concrete Implementation of Naive Bayes Hyperparams."""
from __future__ import annotations

from src.base.hyperparams import Hyperparams


class NaivesBayesHyperparams(Hyperparams):
    random_state: int
    num_classes: int
